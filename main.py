from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"service": "cinemind-api-v5.1"}

@app.get("/health")
def health():
    return {"ok": True, "service": "cinemind-api-v5.1"}

def clamp01(x):
    return float(max(0.0, min(1.0, x)))

def clamp100(x):
    return float(max(0.0, min(100.0, x)))

def safe_resize_for_speed(img, max_dim=1280):
    h, w = img.shape[:2]; m = max(h, w)
    if m <= max_dim: return img
    s = max_dim / m
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def central_roi(gray, frac=0.45):
    h, w = gray.shape[:2]; ch = int(h*frac); cw = int(w*frac)
    y0 = (h-ch)//2; x0 = (w-cw)//2
    return gray[y0:y0+ch, x0:x0+cw], (x0, y0, cw, ch)

def outer_ring_mask(h, w, inner_frac=0.55):
    mask = np.ones((h,w), dtype=np.uint8)
    ch = int(h*inner_frac); cw = int(w*inner_frac)
    y0=(h-ch)//2; x0=(w-cw)//2; mask[y0:y0+ch, x0:x0+cw]=0
    return mask


# ── Scene Type Detection ──────────────────────────────────────────────────────

def detect_scene_type(image_bgr) -> dict:
    """
    'subject' = person/face in frame  |  'ambient' = no person detected

    Uses Haar face detection + skin ratio fallback (>5% = person present).
    This drives: whether skin metric is computed, cinematography weighting,
    suggestion logic, and final score weighting.
    """
    img = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
    has_face = len(faces) > 0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,15,50],dtype=np.uint8), np.array([25,255,255],dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([170,15,50],dtype=np.uint8), np.array([180,255,255],dtype=np.uint8))
    skin_ratio = float(np.count_nonzero(cv2.bitwise_or(m1,m2))) / (img.shape[0]*img.shape[1])

    has_subject = has_face or skin_ratio > 0.05
    return {
        "scene_type": "subject" if has_subject else "ambient",
        "has_face": bool(has_face),
        "skin_ratio": round(skin_ratio, 4),
    }


# ── Exposure ──────────────────────────────────────────────────────────────────

def analyze_exposure(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    mean  = float(np.mean(gray))
    p5    = float(np.percentile(gray,5));  p25 = float(np.percentile(gray,25))
    p50   = float(np.percentile(gray,50)); p75 = float(np.percentile(gray,75))
    p95   = float(np.percentile(gray,95)); p99 = float(np.percentile(gray,99))
    hl_hard = float(np.mean(gray>=250)); hl_soft = float(np.mean(gray>=230))
    sh_hard = float(np.mean(gray<=5));   sh_soft = float(np.mean(gray<=30))

    state = "ok"
    if hl_hard>0.005 or hl_soft>0.02 or p75>210 or p95>225: state="overexposed"
    elif sh_hard>0.01 or sh_soft>0.03 or p25<30 or p50<60:   state="underexposed"

    score = 100.0
    score -= min(50.0, hl_hard*5000); score -= min(30.0, hl_soft*600)
    score -= min(40.0, sh_hard*4000); score -= min(20.0, sh_soft*400)
    score -= min(15.0, abs(p50-118)*0.12)
    return clamp100(score), {
        "mean":round(mean,2),"p5":round(p5,2),"p25":round(p25,2),"p50":round(p50,2),
        "p75":round(p75,2),"p95":round(p95,2),"p99":round(p99,2),
        "highlight_clip":round(hl_hard,6),"highlight_soft":round(hl_soft,6),
        "shadow_clip":round(sh_hard,6),"shadow_soft":round(sh_soft,6),"state":state,
    }


# ── Contrast ─────────────────────────────────────────────────────────────────

def analyze_contrast(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    std  = float(np.std(gray))
    p5   = float(np.percentile(gray,5)); p95 = float(np.percentile(gray,95))
    usable = p95-p5; hl_soft = float(np.mean(gray>=230))
    score = clamp100(100-abs(usable-200)*0.5) - min(30, hl_soft*400)
    return clamp100(score), {"std":round(std,3),"usable_spread":round(usable,2),"p5":round(p5,2),"p95":round(p95,2)}


# ── Color Balance ─────────────────────────────────────────────────────────────

def analyze_color_balance(image_bgr):
    b_ch,g_ch,r_ch = cv2.split(image_bgr)
    rm=float(np.mean(r_ch)); gm=float(np.mean(g_ch)); bm=float(np.mean(b_ch))
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_mean = float(np.mean(lab[:,:,1]-128)); b_lab = float(np.mean(lab[:,:,2]-128))
    lab_cast = float(np.sqrt(a_mean**2+b_lab**2)); rgb_imb = float(np.std([rm,gm,bm]))
    cyan_proxy=(gm+bm)/2-rm; g_dom=gm-rm; b_dom=bm-rm

    temp="neutral"
    if cyan_proxy>8 and g_dom>10: temp="cool/cyan"
    elif b_dom>12 and g_dom<5:    temp="cool"
    elif b_lab<-5:                 temp="cool"
    elif b_lab>5 or (rm-bm>12 and rm-gm>8): temp="warm"

    tint="neutral"
    if a_mean<-5: tint="green"
    elif a_mean>5: tint="magenta"

    eff = max(lab_cast, rgb_imb*1.5)
    return clamp100(100-eff*3.5), {
        "r_mean":round(rm,2),"g_mean":round(gm,2),"b_mean":round(bm,2),
        "lab_a_mean":round(a_mean,3),"lab_b_mean":round(b_lab,3),
        "lab_cast":round(lab_cast,3),"rgb_imbalance":round(rgb_imb,3),
        "effective_cast":round(eff,3),"cyan_proxy":round(cyan_proxy,3),
        "temperature":temp,"tint":tint,
    }


# ── Skin (subject scenes only) ────────────────────────────────────────────────

def analyze_skin(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    m1=cv2.inRange(hsv,np.array([0,15,50],dtype=np.uint8),np.array([25,255,255],dtype=np.uint8))
    m2=cv2.inRange(hsv,np.array([170,15,50],dtype=np.uint8),np.array([180,255,255],dtype=np.uint8))
    mask=cv2.bitwise_or(m1,m2)
    cnt=int(np.count_nonzero(mask)); ratio=float(cnt/(image_bgr.shape[0]*image_bgr.shape[1]+1e-6))
    if cnt<80 or ratio<0.008: return 50.0,{"skin_detected":False,"skin_ratio":round(ratio,6)}
    b_ch,g_ch,r_ch=cv2.split(image_bgr)
    rm=float(np.mean(r_ch[mask>0])); gm=float(np.mean(g_ch[mask>0])); bm=float(np.mean(b_ch[mask>0]))
    dev=float(abs(rm-gm)+abs(rm-bm)); score=clamp100(100-dev*0.45)
    temp="neutral"
    if rm>bm+18: temp="warm"
    elif bm>rm+18: temp="cool"
    return score,{"skin_detected":True,"skin_ratio":round(ratio,6),"r_mean":round(rm,2),"g_mean":round(gm,2),"b_mean":round(bm,2),"temperature":temp,"deviation":round(dev,3)}


# ── Noise ─────────────────────────────────────────────────────────────────────

def analyze_noise(image_bgr):
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    ns=float(np.std(gray.astype(np.float32)-blur.astype(np.float32)))
    snr=float(np.mean(gray))/(ns+1e-6)
    return clamp100(100-ns*2.5),{"noise_std":round(ns,4),"snr":round(snr,3)}


# ── Sharpness ─────────────────────────────────────────────────────────────────

def analyze_sharpness(image_bgr):
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    ls=float(np.std(cv2.Laplacian(gray,cv2.CV_64F)))
    score=clamp100(math.log1p(ls)/math.log1p(80)*100)
    state="sharp"
    if ls<10: state="very_soft"
    elif ls<20: state="soft"
    elif ls<40: state="moderate"
    return score,{"laplacian_std":round(ls,3),"state":state}


# ── Cinematography ────────────────────────────────────────────────────────────

def analyze_cinematography(image_bgr, scene_type: str):
    """
    Subject scenes: full weighting incl. subject_separation + background_blur.
    Ambient scenes: those metrics are excluded (not meaningful without a subject).
    Re-weighted to lighting_depth 40% / layer_complexity 35% / directionality 25%.
    """
    img=safe_resize_for_speed(image_bgr); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); h,w=gray.shape[:2]
    center,(x0,y0,cw,ch)=central_roi(gray); om=outer_ring_mask(h,w)
    op=gray[om>0]; cs=float(np.std(center)); os=float(np.std(op)) if op.size>0 else float(np.std(gray))
    subject_separation=clamp01((cs-os+15)/40)*100
    lap=np.abs(cv2.Laplacian(gray,cv2.CV_64F))
    cl=lap[y0:y0+ch,x0:x0+cw]; ol=lap[om>0]
    cs2=float(np.mean(cl)); os2=float(np.mean(ol)) if ol.size>0 else float(np.mean(lap))
    background_blur=clamp01((1.10-(os2+1e-6)/(cs2+1e-6))/(1.10-0.35))*100
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag_mean=float(np.mean(cv2.magnitude(gx,gy)))
    spread=max(0.,float(np.percentile(center,90)-np.percentile(center,10)))
    lighting_depth=clamp100((clamp01((mag_mean-6)/18)*0.45+clamp01((spread-30)/80)*0.55)*100)
    thresh=float(np.percentile(gray,95)); bm=(gray>=thresh).astype(np.uint8)
    ys,xs=np.where(bm>0); boff=0.0
    if xs.size>=50:
        dx=(float(np.mean(xs))-w/2)/(w/2); dy=(float(np.mean(ys))-h/2)/(h/2)
        boff=float(np.sqrt(dx**2+dy**2))
    directionality=clamp100(35+clamp01(boff/0.6)*65)
    edges=cv2.Canny(gray,60,160)
    mc=np.zeros((h,w),dtype=np.uint8); mc[y0:y0+ch,x0:x0+cw]=1
    mm=np.ones((h,w),dtype=np.uint8); mm[mc>0]=0; mm[om>0]=0
    def ed(m):
        p=int(np.count_nonzero(m)); return 0. if p<=0 else float(np.count_nonzero(edges[m>0])/p)
    p=np.array([ed(mc),ed(mm),ed(om)],dtype=np.float32); p/=(float(np.sum(p))+1e-6)
    entropy=float(-(p*np.log(p+1e-6)).sum()); layer_complexity=clamp100(clamp01(entropy/1.05)*100)

    if scene_type=="ambient":
        score=clamp100(lighting_depth*0.40+layer_complexity*0.35+directionality*0.25)
    else:
        score=clamp100(subject_separation*0.26+lighting_depth*0.26+background_blur*0.18+layer_complexity*0.16+directionality*0.14)

    m={"lighting_depth":round(lighting_depth,2),"layer_complexity":round(layer_complexity,2),"directionality":round(directionality,2),"bright_offset":round(boff,4),"bg_sharpness":round(os2,4),"subject_sharpness":round(cs2,4)}
    if scene_type=="subject":
        m["subject_separation"]=round(subject_separation,2); m["background_blur"]=round(background_blur,2)
        m["center_contrast_std"]=round(cs,3); m["outer_contrast_std"]=round(os,3)
    return score,m


# ── Composition ───────────────────────────────────────────────────────────────

def analyze_composition(image_bgr):
    img=safe_resize_for_speed(image_bgr); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); h,w=gray.shape[:2]
    edges=cv2.Canny(gray,60,160); ys,xs=np.where(edges>0)
    cx=int(np.mean(xs)) if xs.size>=200 else w//2; cy=int(np.mean(ys)) if xs.size>=200 else h//2
    thirds=[(w/3,h/3),(2*w/3,h/3),(w/3,2*h/3),(2*w/3,2*h/3)]
    dmin=float(min([np.hypot(cx-tx,cy-ty) for tx,ty in thirds])); diag=float(np.hypot(w,h))+1e-6
    rot=clamp100((1-(dmin/(0.55*diag)))*100)
    le=float(np.mean(edges[:,:w//2]>0)); re=float(np.mean(edges[:,w//2:]>0))
    bal=clamp100((1-min(1,abs(le-re)/0.08))*100)
    ed=float(np.mean(edges>0)); ns=clamp100((1-min(1,ed/0.12))*100)
    tilt_deg=0.0
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=80,minLineLength=80,maxLineGap=10)
    if lines is not None:
        ah=[]; wh=[]
        for l in lines[:,0]:
            x1,y1,x2,y2=l; length=float(np.hypot(x2-x1,y2-y1))
            if length<60: continue
            angle=np.degrees(np.arctan2(y2-y1,x2-x1))
            if angle>90: angle-=180
            if angle<-90: angle+=180
            if abs(angle)<=25: ah.append(angle); wh.append(length)
        if len(ah)>=2: tilt_deg=float(np.average(np.array(ah),weights=np.array(wh)))
        elif len(ah)==1: tilt_deg=float(ah[0])
    ts=clamp100(100-abs(tilt_deg)*8)
    score=clamp100(rot*0.35+bal*0.25+ns*0.25+ts*0.15)
    return score,{"subject_position":{"x":cx,"y":cy},"edge_density":round(ed,6),"tilt_deg":round(float(tilt_deg),3),"tilt_score":round(float(ts),1),"rule_of_thirds":round(float(rot),2),"balance":round(float(bal),2),"negative_space":round(float(ns),2)}


# ── Suggestions ───────────────────────────────────────────────────────────────

def build_suggestions(exposure_m, color_m, cine_m, comp_m, sharpness_m, scene_info):
    s=[]; scene_type=scene_info.get("scene_type","subject")

    state=exposure_m.get("state","ok")
    hl_hard=float(exposure_m.get("highlight_clip",0)); hl_soft=float(exposure_m.get("highlight_soft",0))
    sh_hard=float(exposure_m.get("shadow_clip",0));   sh_soft=float(exposure_m.get("shadow_soft",0))
    if state=="overexposed":
        if hl_hard>0.02: msg="Highlights are heavily clipped. Lower exposure ~1–1.5 stops or reduce key light to recover texture in whites."
        elif hl_soft>0.05: msg="Image looks washed out / over-bright. Lower exposure ~0.5–1 stop to restore depth and contrast in the highlights."
        else: msg="Image is slightly overexposed. Reduce exposure ~0.5 stop or protect highlights with a gentle S-curve in grading."
        s.append({"category":"Exposure","priority":"high","message":msg})
    elif state=="underexposed":
        if sh_hard>0.02: msg="Blacks are severely crushed. Lift exposure ~1–1.5 stops or add fill light to recover shadow detail."
        elif sh_soft>0.05: msg="Image is underexposed with heavy shadows. Increase exposure ~0.5–1 stop or add soft fill to lift midtones."
        else: msg="Image looks slightly underexposed. Increase exposure ~0.5 stop or raise midtones gently in post."
        s.append({"category":"Exposure","priority":"high","message":msg})

    eff=float(color_m.get("effective_cast",0)); temp=color_m.get("temperature","neutral"); tint=color_m.get("tint","neutral")
    if eff>=8.0:
        parts=[]
        if temp=="cool/cyan": parts.append("cyan/teal")
        elif temp=="cool":    parts.append("cool/blue")
        elif temp=="warm":    parts.append("warm/yellow-orange")
        if tint=="green":     parts.append("green")
        elif tint=="magenta": parts.append("magenta")
        label=" + ".join(parts) if parts else "color"
        s.append({"category":"Color","priority":"high" if eff>=12 else "medium","message":f"Noticeable {label} cast detected. Correct white balance (Temp/Tint sliders) before applying any creative look."})

    ls=float(sharpness_m.get("laplacian_std",50)); sh_state=sharpness_m.get("state","sharp")
    if sh_state in ("very_soft","soft"):
        msg="Image appears very soft or hazy. Check for lens fog, diffusion filter, or significant motion blur. Re-shoot if critical." if ls<8 else "Image looks slightly soft. Ensure subject is in focus and try a faster shutter speed to reduce motion blur."
        s.append({"category":"Sharpness","priority":"medium","message":msg})

    if scene_type=="subject":
        sep=float(cine_m.get("subject_separation",50))
        if sep<35: s.append({"category":"Cinematography","priority":"medium","message":"Subject separation is low. Increase subject–background distance, simplify the background, or use a longer focal length to compress and isolate the subject."})
    else:
        layer=float(cine_m.get("layer_complexity",50))
        if layer<35: s.append({"category":"Cinematography","priority":"low","message":"The scene looks visually flat. Try adding foreground elements, leading lines, or varying depth to create more visual interest."})

    td=float(comp_m.get("tilt_deg",0))
    if abs(td)>=2.5: s.append({"category":"Composition","priority":"medium","message":f"Horizon/lines appear tilted (~{abs(td):.1f}°). Level the shot in-camera or correct rotation in post for a cleaner, more professional look."})
    return s


# ── Main Endpoint ─────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents=await file.read()
    image=cv2.imdecode(np.frombuffer(contents,np.uint8), cv2.IMREAD_COLOR)
    if image is None: return {"ok":False,"error":"Invalid image"}

    scene_info=detect_scene_type(image); scene_type=scene_info["scene_type"]

    exposure_score,  exposure_metrics  = analyze_exposure(image)
    contrast_score,  contrast_metrics  = analyze_contrast(image)
    color_score,     color_metrics     = analyze_color_balance(image)
    noise_score,     noise_metrics     = analyze_noise(image)
    sharpness_score, sharpness_metrics = analyze_sharpness(image)
    cine_score,      cine_metrics      = analyze_cinematography(image, scene_type)
    comp_score,      comp_metrics      = analyze_composition(image)

    skin_score=skin_metrics=None
    if scene_type=="subject":
        skin_score,skin_metrics=analyze_skin(image)

    suggestions=build_suggestions(exposure_metrics,color_metrics,cine_metrics,comp_metrics,sharpness_metrics,scene_info)

    if scene_type=="subject":
        cinematic_score=int(exposure_score*0.22+contrast_score*0.12+color_score*0.16+(skin_score or 50)*0.10+noise_score*0.06+sharpness_score*0.08+cine_score*0.14+comp_score*0.12)
    else:
        cinematic_score=int(exposure_score*0.22+contrast_score*0.16+color_score*0.20+noise_score*0.06+sharpness_score*0.08+cine_score*0.16+comp_score*0.12)

    breakdown={"exposure":round(exposure_score,1),"contrast":round(contrast_score,1),"color":round(color_score,1)}
    if scene_type=="subject" and skin_score is not None: breakdown["skin"]=round(skin_score,1)
    breakdown.update({"noise":round(noise_score,1),"sharpness":round(sharpness_score,1),"cinematography":round(cine_score,1),"composition":round(comp_score,1)})

    metrics={"scene":scene_info,"exposure":exposure_metrics,"contrast":contrast_metrics,"color":color_metrics,"noise":noise_metrics,"sharpness":sharpness_metrics,"cinematography":cine_metrics,"composition":comp_metrics}
    if scene_type=="subject" and skin_metrics: metrics["skin"]=skin_metrics

    return {"ok":True,"score":cinematic_score,"scene_type":scene_type,"breakdown":breakdown,"metrics":metrics,"suggestions":suggestions}
