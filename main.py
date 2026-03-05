from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2, math, os, base64, json as _json_mod
import httpx

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def build_vision_prompt(opencv_data: dict) -> str:
    scene   = opencv_data.get("scene_type", "unknown")
    lum     = opencv_data.get("luminosity_type", "unknown")
    bokeh   = opencv_data.get("bokeh_ratio", 0)
    light   = opencv_data.get("light_direction", "unknown")
    harmony = opencv_data.get("harmony", "unknown")
    sat     = opencv_data.get("sat_mean", 0)
    spread  = opencv_data.get("saliency_spread", 0.5)
    faces   = len(opencv_data.get("faces", []))
    t       = opencv_data.get("tonal", {})
    p50     = t.get(50, 0)
    dr      = t.get("dynamic_range", 0)
    tech_problems = opencv_data.get("_known_issues", [])
    intentional   = opencv_data.get("_intentional", [])
    issues_str    = ", ".join(tech_problems) if tech_problems else "none"
    intent_str    = (" | ".join(intentional)) if intentional else "none"
    return (
        "You are a professional video and photography consultant with deep knowledge of "
        "cinematography, color, composition, and visual storytelling across commercial "
        "production, branded content, streaming series, music videos, documentary, and "
        "social media video.\n\n"
        "You are analyzing a Rec.709 preview frame from Apple Log footage shot on an "
        "iPhone or consumer camera.\n\n"
        "YOUR REFERENCE STANDARD: published professional content — TV commercials, "
        "branded social media campaigns, streaming series, editorial photography, music "
        "videos. Not Hollywood blockbusters, but the polished professional work you see "
        "from competent production companies every day.\n\n"
        "SCORING SCALE:\n"
        "- 30-50: Unintentional, no visual awareness, significant technical errors\n"
        "- 51-65: Some awareness and effort, but execution has notable problems or lacks clear intent\n"
        "- 66-75: Solid work — clear subject, decent exposure, intentional choices, publishable with minor fixes\n"
        "- 76-85: Strong professional-looking result — good light, clear story, well composed\n"
        "- 86-95: Excellent — stands out even among professional content\n"
        "- 96-100: Exceptional, outstanding even by broadcast/streaming standards\n\n"
        f"Technical measurements already confirmed:\n"
        f"- Scene: {scene} | Luminosity: {lum} | Faces: {faces}\n"
        f"- Exposure p50={p50:.0f} | Dynamic range={dr:.0f}\n"
        f"- Bokeh ratio={bokeh:.2f} (>1.8=isolation, >3.0=strong)\n"
        f"- Light direction: {light}\n"
        f"- Color saturation={sat:.2f} | harmony={harmony}\n"
        f"- Saliency spread={spread:.2f} (lower=more focused)\n"
        f"- Technical problems already confirmed: {issues_str}\n"
        f"- Confirmed intentional creative choices (do NOT flag as problems): {intent_str}\n\n"
        "STRICT RULES:\n"
        "- If a technical problem is confirmed above, do NOT list that area as a strength\n"
        "- Do NOT invent strengths — only list what is genuinely and clearly working\n"
        "- Do NOT contradict yourself between strengths and concerns\n"
        "- Concerns must be clearly visible and impactful — do not nitpick\n\n"
        "Respond ONLY with valid JSON, no markdown, no extra text:\n"
        "{\n"
        '  \"creative_score\": <integer 0-100>,\n'
        '  \"overall_read\": \"<one honest sentence: what is this shot and does it work?>\",\n'
        '  \"strengths\": [{{\"label\": \"<max 5 words>\", \"note\": \"<one sentence>\"}}],\n'
        '  \"concerns\": [{{\"label\": \"<max 5 words>\", \"note\": \"<one sentence>\"}}],\n'
        '  \"grade_intent\": \"<neutral|warm|cool|desaturated|stylized>\",\n'
        '  \"color_cast_detail\": \"<precise description of visible cast e.g. magenta in midtones, or empty>\",\n'
        '  \"lighting_read\": \"<one sentence on lighting quality and intent>\"\n'
        "}"
    )


async def analyze_vision(image_bgr, opencv_data: dict) -> dict:
    if not ANTHROPIC_API_KEY:
        return {}
    try:
        img = image_bgr
        h, w = img.shape[:2]
        if max(h, w) > 1024:
            s = 1024 / max(h, w)
            img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        b64 = base64.b64encode(buf.tobytes()).decode()
        prompt = build_vision_prompt(opencv_data)
        payload = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 700,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt}
            ]}]
        }
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json=payload
            )
        if resp.status_code != 200:
            return {}
        text = resp.json()["content"][0]["text"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return _json_mod.loads(text)
    except Exception:
        return {}

def merge_vision_into_creative(creative: dict, vision: dict, scene_type: str) -> dict:
    """
    Vision API is primary source for creative score and elements.
    Context boosts applied for intentional cinematic choices Vision may undervalue.
    Weighting: subject 50/50, ambient 70/30.
    Severe penalty applied when multiple confirmed technical problems exist.
    """
    if not vision:
        return creative

    vision_score = float(vision.get("creative_score", 50))
    opencv_score = float(creative.get("creative_score", 50))

    # How many confirmed technical problems exist?
    known_issues = creative.get("_known_issues_count", 0)

    # Context-aware boost for intentional cinematic choices
    opencv_elements = creative.get("elements_detected", [])
    opencv_signals  = {e.get("signal","") for e in opencv_elements}
    boost = 0.0
    if "low_key_portrait" in opencv_signals:      boost += 6.0
    if any("bokeh" in s for s in opencv_signals):  boost += 5.0
    if any("leading" in s for s in opencv_signals): boost += 3.0
    boost = min(boost, 12.0)
    # No boost if scene is technically compromised
    if known_issues >= 2:
        boost = 0.0
    vision_score_adj = min(100.0, vision_score + boost)

    if scene_type == "subject":
        final_creative = vision_score_adj * 0.70 + opencv_score * 0.30
    else:
        final_creative = vision_score_adj * 0.50 + opencv_score * 0.50

    # Cap elements based on severity of technical problems
    # 0-1 issues: up to 4 elements
    # 2 issues:   up to 1 element (and only if genuinely unrelated to the problems)
    # 3+ issues:  no elements — scene is too compromised to highlight positives
    if known_issues >= 3:
        max_elements = 0
    elif known_issues == 2:
        max_elements = 1
    else:
        max_elements = 4

    vision_strengths = vision.get("strengths", [])
    vision_concerns  = vision.get("concerns", [])

    elements = [{"signal": f"vision_{i}", "label": s["label"], "note": s["note"]}
                for i, s in enumerate(vision_strengths[:max_elements])]
    concerns  = [{"signal": f"vision_c_{i}", "label": c["label"], "note": c["note"]}
                for i, c in enumerate(vision_concerns[:2])]

    # Add OpenCV signals only if we still have room and scene isn't too compromised
    if known_issues < 2:
        vision_labels_low = {e["label"].lower() for e in elements}
        for e in opencv_elements:
            if not any(kw in e["label"].lower() for kw in ["bokeh","leading lines","rule","low-key","directional"]):
                continue
            if not any(kw in vision_labels_low for kw in e["label"].lower().split()):
                elements.append(e)
            if len(elements) >= max_elements:
                break

    result = dict(creative)
    result["elements_detected"] = elements[:max_elements]
    result["concerns"]          = concerns[:2]
    result["creative_score"]    = float(max(0, min(100, final_creative)))
    result["vision"] = {
        "creative_score":     vision_score,
        "creative_score_adj": round(vision_score_adj, 1),
        "overall_read":       vision.get("overall_read", ""),
        "grade_intent":       vision.get("grade_intent", ""),
        "lighting_read":      vision.get("lighting_read", ""),
        "color_cast_detail":  vision.get("color_cast_detail", ""),
    }
    return result

@app.get("/")
def root(): return {"service": "cinemind-api-v6"}

@app.get("/health")
def health(): return {"ok": True, "service": "cinemind-api-v6"}

# ── utils ──────────────────────────────────────────────────────────────────

def clamp(x, lo=0., hi=100.): return float(max(lo, min(hi, x)))
def clamp01(x): return float(max(0., min(1., x)))
def lap_std(patch): return float(np.std(cv2.Laplacian(patch.astype(np.uint8), cv2.CV_64F)))

def safe_resize(img, max_dim=1280):
    h, w = img.shape[:2]; m = max(h, w)
    if m <= max_dim: return img
    s = max_dim / m
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def central_roi(gray, frac=0.45):
    h, w = gray.shape[:2]; ch, cw = int(h*frac), int(w*frac)
    y0, x0 = (h-ch)//2, (w-cw)//2
    return gray[y0:y0+ch, x0:x0+cw], (x0, y0, cw, ch)

def outer_mask(h, w, inner_frac=0.55):
    mask = np.ones((h, w), dtype=np.uint8)
    ch, cw = int(h*inner_frac), int(w*inner_frac)
    y0, x0 = (h-ch)//2, (w-cw)//2
    mask[y0:y0+ch, x0:x0+cw] = 0
    return mask

# ── scene context ──────────────────────────────────────────────────────────

def build_scene_context(image_bgr):
    img  = safe_resize(image_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # tonal distribution
    pct = {p: float(np.percentile(gray, p)) for p in [1,5,10,25,50,75,90,95,99]}
    dark   = float(np.mean(gray < 64))
    mid    = float(np.mean((gray >= 64) & (gray <= 192)))
    bright = float(np.mean(gray > 192))
    dr     = pct[99] - pct[1]
    if   dark > 0.40 and bright < 0.20:             lum = "LOW-KEY"
    elif bright > 0.40 and dark < 0.15:             lum = "HIGH-KEY"
    elif dr > 150 and dark > 0.15 and bright > 0.10: lum = "CONTRASTY"
    else:                                             lum = "BALANCED"
    tonal = {**pct, "dark_mass": round(dark,4), "mid_mass": round(mid,4),
             "bright_mass": round(bright,4), "dynamic_range": round(dr,1)}

    # face detection
    fc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    raw = fc.detectMultiScale(gray, 1.1, 5, minSize=(40,40))
    faces = [tuple(map(int,f)) for f in raw] if len(raw) > 0 else []
    largest_face = max(faces, key=lambda f: f[2]*f[3]) if faces else None

    # skin ratio
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0,15,50],   dtype=np.uint8), np.array([25,255,255], dtype=np.uint8))
    m2 = cv2.inRange(hsv, np.array([170,15,50],  dtype=np.uint8), np.array([180,255,255],dtype=np.uint8))
    skin_ratio = float(np.count_nonzero(cv2.bitwise_or(m1,m2))) / (h*w)
    scene_type = "subject" if (len(faces)>0 or skin_ratio>0.05) else "ambient"

    # bokeh ratio
    cy2, cx2 = h//2, w//2; bh, bw = h//3, w//3
    cp = gray[cy2-bh//2:cy2+bh//2, cx2-bw//2:cx2+bw//2]
    cs = min(h,w)//6
    corners = [gray[:cs,:cs], gray[:cs,w-cs:], gray[h-cs:,:cs], gray[h-cs:,w-cs:]]
    c_lap   = lap_std(cp)
    crn_lap = float(np.mean([lap_std(c) for c in corners]))
    bokeh_ratio    = c_lap / (crn_lap + 1e-6)
    bokeh_detected = bokeh_ratio > 1.8

    # light direction from bright centroid
    thresh = float(np.percentile(gray, 90))
    ys, xs = np.where(gray >= thresh)
    if len(xs) >= 50:
        lcx = float(np.mean(xs))/w; lcy = float(np.mean(ys))/h
        if   lcx < 0.35 and lcy < 0.4: ld = "top-left"
        elif lcx > 0.65 and lcy < 0.4: ld = "top-right"
        elif lcy < 0.30:                ld = "top"
        elif lcy > 0.70:                ld = "bottom"
        elif lcx < 0.35:                ld = "left"
        elif lcx > 0.65:                ld = "right"
        else:                           ld = "frontal"
    else:
        lcx, lcy, ld = 0.5, 0.5, "unknown"

    # color harmony
    sat_ch   = hsv[:,:,1]
    sat_mean = float(np.mean(sat_ch)) / 255.0
    mask_sat = sat_ch > 60
    dom_hues = []; harmony = "monochromatic"
    if np.count_nonzero(mask_sat) > 500:
        hues_px = hsv[:,:,0][mask_sat].astype(np.int32)
        hist = np.bincount(hues_px, minlength=180).astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        peaks = sorted([(i*2, hist[i]) for i in range(90) if hist[i]>0.04], key=lambda x: -x[1])
        dom_hues = [p[0] for p in peaks[:3]]
        if len(dom_hues) >= 2:
            diff = abs(dom_hues[0]-dom_hues[1]); diff = min(diff, 360-diff)
            if   diff > 150:          harmony = "complementary"
            elif diff < 40:           harmony = "analogous"
            elif 100 < diff < 140:    harmony = "split-complementary"
            else:                     harmony = "complex"

    # saliency (spectral residual)
    small    = cv2.resize(gray, (64,64)).astype(np.float32)
    fft      = np.fft.fft2(small)
    logamp   = np.log(np.abs(fft)+1e-6)
    blur_log = cv2.GaussianBlur(logamp.astype(np.float32),(3,3),0)
    residual = logamp - blur_log
    sal_fft  = np.exp(residual)*np.exp(1j*np.angle(fft))
    sal_map  = np.abs(np.fft.ifft2(sal_fft))**2
    sal_map  = cv2.GaussianBlur(sal_map.astype(np.float32),(5,5),0)
    sal_map  = cv2.resize(sal_map,(w,h))
    mn, mx   = sal_map.min(), sal_map.max()
    sal_norm = (sal_map-mn)/(mx-mn+1e-6)
    sm = sal_norm > float(np.percentile(sal_norm,85))
    ys2, xs2 = np.where(sm)
    sal_cx  = float(np.mean(xs2)/w) if len(xs2)>10 else 0.5
    sal_cy  = float(np.mean(ys2)/h) if len(xs2)>10 else 0.5
    sal_spr = float(np.std(xs2)/w + np.std(ys2)/h) if len(xs2)>10 else 0.5

    # lines
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,minLineLength=60,maxLineGap=15)
    diag = horiz = vert = 0
    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2=l; a=abs(np.degrees(np.arctan2(y2-y1,x2-x1)))
            if   20<a<70 or 110<a<160: diag+=1
            elif a<15 or a>165:        horiz+=1
            elif 75<a<105:             vert+=1

    return {
        "scene_type": scene_type, "luminosity_type": lum,
        "faces": faces, "largest_face": largest_face, "skin_ratio": round(skin_ratio,4),
        "bokeh_ratio": round(bokeh_ratio,3), "bokeh_detected": bokeh_detected,
        "light_direction": ld, "light_cx": round(lcx,3), "light_cy": round(lcy,3),
        "harmony": harmony, "dom_hues": [round(h) for h in dom_hues],
        "sat_mean": round(sat_mean,4),
        "saliency_cx": round(sal_cx,3), "saliency_cy": round(sal_cy,3),
        "saliency_spread": round(sal_spr,3),
        "diag_lines": diag, "horiz_lines": horiz, "vert_lines": vert,
        "tonal": tonal, "_img": img,
    }

# ── exposure ───────────────────────────────────────────────────────────────

def analyze_exposure(image_bgr, ctx):
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    t       = ctx["tonal"]
    hl_hard = float(np.mean(gray>=250)); hl_soft = float(np.mean(gray>=230))
    sh_hard = float(np.mean(gray<=5));   sh_soft = float(np.mean(gray<=30))
    lum     = ctx["luminosity_type"]
    state   = "ok"
    if lum == "LOW-KEY":
        if hl_hard>0.01  or hl_soft>0.05:          state="overexposed"
        elif sh_hard>0.03 or t[25]<8:              state="underexposed"
    elif lum == "HIGH-KEY":
        if hl_hard>0.003 or hl_soft>0.015 or t[75]>215 or t[95]>228: state="overexposed"
        elif sh_soft>0.02 or t[50]<80:             state="underexposed"
    elif lum == "CONTRASTY":
        if hl_hard>0.01  or hl_soft>0.04:          state="overexposed"
        elif sh_hard>0.02:                          state="underexposed"
    else:
        if hl_hard>0.005 or hl_soft>0.02 or t[75]>210 or t[95]>225: state="overexposed"
        elif sh_hard>0.01 or sh_soft>0.03 or t[25]<30 or t[50]<60:  state="underexposed"
    score = 100.
    score -= clamp(hl_hard*5000,0,50); score -= clamp(hl_soft*400,0,25)
    score -= clamp(sh_hard*4000,0,40); score -= clamp(sh_soft*300,0,15)
    if lum in ("BALANCED","HIGH-KEY"): score -= clamp(abs(t[50]-118)*0.10,0,12)
    return clamp(score), {"state":state,"luminosity_type":lum,
        "p25":round(t[25],1),"p50":round(t[50],1),"p75":round(t[75],1),"p95":round(t[95],1),
        "highlight_clip":round(hl_hard,6),"highlight_soft":round(hl_soft,6),
        "shadow_clip":round(sh_hard,6),"shadow_soft":round(sh_soft,6)}

# ── contrast ───────────────────────────────────────────────────────────────

def analyze_contrast(image_bgr, ctx):
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    std     = float(np.std(gray))
    t       = ctx["tonal"]; lum = ctx["luminosity_type"]
    usable  = t[95]-t[5]
    hl_soft = float(np.mean(gray>=230))
    target  = {"LOW-KEY":160,"CONTRASTY":200,"HIGH-KEY":130,"BALANCED":190}.get(lum,190)
    penalty = min(25.,hl_soft*400) if lum in ("BALANCED","HIGH-KEY") else 0
    return clamp(clamp(100-abs(usable-target)*0.5)-penalty), {
        "std":round(std,3),"usable_spread":round(usable,2),"target_spread":target,"luminosity_type":lum}

# ── color ──────────────────────────────────────────────────────────────────

def analyze_color_balance(image_bgr, ctx):
    b_ch,g_ch,r_ch = cv2.split(image_bgr)
    rm=float(np.mean(r_ch)); gm=float(np.mean(g_ch)); bm=float(np.mean(b_ch))
    lab = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
    am  = float(np.mean(lab[:,:,1]-128)); bm2=float(np.mean(lab[:,:,2]-128))
    lab_cast = float(np.sqrt(am**2+bm2**2)); rgb_imb=float(np.std([rm,gm,bm]))
    eff_cast = max(lab_cast,rgb_imb*1.5)
    sat_mean = ctx["sat_mean"]; lum = ctx["luminosity_type"]
    cyan_proxy=(gm+bm)/2-rm; g_dom=gm-rm; b_dom=bm-rm
    temp="neutral"
    if cyan_proxy>8 and g_dom>10:           temp="cool/cyan"
    elif b_dom>12 and g_dom<5:              temp="cool"
    elif bm2<-5:                            temp="cool"
    elif bm2>5 or (rm-bm>12 and rm-gm>8):  temp="warm"
    tint="neutral"
    if am<-5: tint="green"
    elif am>5: tint="magenta"
    is_cw = (lum=="LOW-KEY" and temp=="warm" and bm2>3 and am>0)
    is_cc = (lum in ("LOW-KEY","CONTRASTY") and temp in ("cool","cool/cyan") and eff_cast<18 and sat_mean>0.15)
    creative_grade = "possible" if (is_cw or is_cc) else "unlikely"
    if sat_mean < 0.12: eff_cast *= 0.5
    return clamp(100-eff_cast*3.5), {
        "r_mean":round(rm,2),"g_mean":round(gm,2),"b_mean":round(bm,2),
        "lab_a_mean":round(am,3),"lab_b_mean":round(bm2,3),
        "lab_cast":round(lab_cast,3),"rgb_imbalance":round(rgb_imb,3),
        "effective_cast":round(eff_cast,3),"cyan_proxy":round(cyan_proxy,3),
        "temperature":temp,"tint":tint,"creative_grade":creative_grade,
        "harmony":ctx["harmony"],"dominant_hues_deg":ctx["dom_hues"]}

# ── skin ───────────────────────────────────────────────────────────────────

def analyze_skin(image_bgr, ctx):
    hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)
    m1=cv2.inRange(hsv,np.array([0,15,50],dtype=np.uint8),np.array([25,255,255],dtype=np.uint8))
    m2=cv2.inRange(hsv,np.array([170,15,50],dtype=np.uint8),np.array([180,255,255],dtype=np.uint8))
    mask=cv2.bitwise_or(m1,m2); cnt=int(np.count_nonzero(mask))
    ratio=float(cnt)/(image_bgr.shape[0]*image_bgr.shape[1]+1e-6)
    if cnt<80 or ratio<0.008: return 50.,{"skin_detected":False,"skin_ratio":round(ratio,6)}
    b_ch,g_ch,r_ch=cv2.split(image_bgr)
    rm=float(np.mean(r_ch[mask>0])); gm=float(np.mean(g_ch[mask>0])); bm=float(np.mean(b_ch[mask>0]))
    dev=float(abs(rm-gm)+abs(rm-bm)); score=clamp(100-dev*0.45)
    temp="neutral"
    if rm>bm+18: temp="warm"
    elif bm>rm+18: temp="cool"
    return score,{"skin_detected":True,"skin_ratio":round(ratio,6),
        "r_mean":round(rm,2),"g_mean":round(gm,2),"b_mean":round(bm,2),"temperature":temp,"deviation":round(dev,3)}

# ── noise ──────────────────────────────────────────────────────────────────

def analyze_noise(image_bgr, ctx):
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    ns=float(np.std(gray.astype(np.float32)-blur.astype(np.float32)))
    snr=float(np.mean(gray))/(ns+1e-6)
    factor=1.5 if ctx["luminosity_type"]=="LOW-KEY" else 2.0
    return clamp(100-ns*factor),{"noise_std":round(ns,4),"snr":round(snr,3)}

# ── sharpness ─────────────────────────────────────────────────────────────

def analyze_sharpness(image_bgr, ctx):
    """
    Luminance-normalised sharpness.
    Face present: uses eye-strip ROI (top 60% of face) — most focus-diagnostic zone.
    Normalises Laplacian by local mean brightness so LOW-KEY scenes compare fairly.
    Thresholds on norm_lap: very_soft<3  soft<7  moderate<14  sharp>=14
    """
    gray=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    img=safe_resize(gray) if gray.shape[1]>1280 else gray
    h,w=img.shape[:2]; lf=ctx["largest_face"]; bokeh=ctx["bokeh_detected"]
    if lf is not None:
        scale=img.shape[1]/image_bgr.shape[1]
        fx,fy,fw,fh=int(lf[0]*scale),int(lf[1]*scale),int(lf[2]*scale),int(lf[3]*scale)
        # Eye strip: top 60% of face height
        eye_roi=img[max(0,fy):min(h,fy+int(fh*0.60)), max(0,fx):min(w,fx+fw)]
        full_roi=img[max(0,fy):min(h,fy+fh), max(0,fx):min(w,fx+fw)]
        roi_mean=float(np.mean(full_roi)) if full_roi.size>0 else 100.
        ls=lap_std(eye_roi) if eye_roi.size>=200 else lap_std(full_roi)
        source="eye_strip" if eye_roi.size>=200 else "face_roi"
        norm_factor=max(0.3, roi_mean/100.)
        ls_norm=ls/norm_factor
        if   ls_norm<3:  state="very_soft"
        elif ls_norm<7:  state="soft"
        elif ls_norm<14: state="moderate"
        else:            state="sharp"
    else:
        ls=lap_std(img); mean_lum=float(np.mean(img))
        ls_norm=ls/max(0.3,mean_lum/100.); source="global"
        if   ls_norm<2:  state="very_soft"
        elif ls_norm<4:  state="soft"
        elif ls_norm<8:  state="moderate"
        else:            state="sharp"
    score=clamp(math.log1p(ls_norm)/math.log1p(60)*100)
    return score,{"laplacian_raw":round(ls,3),"laplacian_norm":round(ls_norm,3),
        "state":state,"source":source,"bokeh_detected":bokeh,"bokeh_ratio":ctx["bokeh_ratio"]}

# ── cinematography ─────────────────────────────────────────────────────────

def analyze_cinematography(image_bgr, ctx):
    img=ctx["_img"]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); h,w=gray.shape[:2]
    scene=ctx["scene_type"]
    center,(x0,y0,cw,ch)=central_roi(gray); om=outer_mask(h,w)
    cs=float(np.std(center)); os_=float(np.std(gray[om>0]))
    subject_sep=clamp01((cs-os_+15)/40)*100
    bokeh_iso=clamp(ctx["bokeh_ratio"]/4.0*100)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag=float(np.mean(cv2.magnitude(gx,gy)))
    spread=max(0.,float(np.percentile(center,90)-np.percentile(center,10)))
    ld_score=clamp((clamp01((mag-6)/18)*0.45+clamp01((spread-30)/80)*0.55)*100)
    ld_bonus={"top-left":15,"top-right":15,"left":15,"right":15,"top":5}.get(ctx["light_direction"],0)
    lighting_depth=clamp(ld_score+ld_bonus)
    edges=cv2.Canny(gray,60,160)
    mc=np.zeros((h,w),dtype=np.uint8); mc[y0:y0+ch,x0:x0+cw]=1
    mm=np.ones((h,w),dtype=np.uint8); mm[mc>0]=0; mm[om>0]=0
    def ed(m):
        p=int(np.count_nonzero(m)); return 0. if p<=0 else float(np.count_nonzero(edges[m>0])/p)
    p=np.array([ed(mc),ed(mm),ed(om)],dtype=np.float32); p/=(float(np.sum(p))+1e-6)
    layer_complexity=clamp(clamp01(float(-(p*np.log(p+1e-6)).sum())/1.05)*100)
    focal=clamp((1.-clamp01(ctx["saliency_spread"]/0.7))*100)
    leading=clamp(min(ctx["diag_lines"],40)/40.*100)
    if scene=="subject":
        score=clamp(bokeh_iso*0.22+lighting_depth*0.28+layer_complexity*0.14+focal*0.18+leading*0.08+subject_sep*0.10)
    else:
        score=clamp(lighting_depth*0.35+layer_complexity*0.25+leading*0.20+focal*0.20)
    m={"lighting_depth":round(lighting_depth,2),"layer_complexity":round(layer_complexity,2),
       "focal_strength":round(focal,2),"leading_lines_score":round(leading,2),
       "light_direction":ctx["light_direction"],"bokeh_isolation":round(bokeh_iso,2),"bokeh_ratio":ctx["bokeh_ratio"]}
    if scene=="subject": m["subject_separation"]=round(subject_sep,2)
    return score,m

# ── composition ────────────────────────────────────────────────────────────

def analyze_composition(image_bgr, ctx):
    img=ctx["_img"]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); h,w=gray.shape[:2]
    edges=cv2.Canny(gray,60,160)
    cx=int(ctx["saliency_cx"]*w); cy=int(ctx["saliency_cy"]*h)
    thirds=[(w/3,h/3),(2*w/3,h/3),(w/3,2*h/3),(2*w/3,2*h/3)]
    dmin=float(min([np.hypot(cx-tx,cy-ty) for tx,ty in thirds]))
    rot=clamp((1.-dmin/(0.55*float(np.hypot(w,h))+1e-6))*100)
    le=float(np.mean(edges[:,:w//2]>0)); re=float(np.mean(edges[:,w//2:]>0))
    bal=clamp((1.-min(1.,abs(le-re)/0.08))*100)
    ed=float(np.mean(edges>0)); neg=clamp((1.-min(1.,ed/0.12))*100)
    conv=clamp(min(ctx["diag_lines"]+ctx["horiz_lines"]*0.3,60)/60.*100)
    focal=clamp((1.-clamp01(ctx["saliency_spread"]/0.7))*100)
    tilt=0.
    _tilt_lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=60,minLineLength=int(w*0.18),maxLineGap=20)
    if _tilt_lines is not None:
        _ta=[]; _tw=[]; _ty=[]
        for l in _tilt_lines[:,0]:
            x1,y1,x2,y2=l; ln=float(np.hypot(x2-x1,y2-y1))
            a=np.degrees(np.arctan2(y2-y1,x2-x1))
            if a>90: a-=180
            if a<-90: a+=180
            if abs(a)<=15: _ta.append(a); _tw.append(ln); _ty.append((y1+y2)/2./h)
        if len(_ta)>=2:
            _arr=np.array(_ta); _warr=np.array(_tw); _mpy=np.array(_ty)
            _tw_total=_warr.sum()
            _bc=np.arange(-15.5,16.5,1.); _bct=(_bc[:-1]+_bc[1:])/2
            _wh=np.zeros(len(_bct))
            for _a,_w in zip(_arr,_warr):
                _i=int(np.argmin(np.abs(_bct-_a))); _wh[_i]+=_w
            _pk=int(np.argmax(_wh)); _pc=_bct[_pk]
            _cm=np.abs(_arr-_pc)<=4.; _cons=_warr[_cm].sum()/(_tw_total+1e-6)
            if _cons>=0.50:
                _ys=float(np.ptp(_mpy[_cm]))
                if _ys>=0.20:
                    tilt=float(np.average(_arr[_cm],weights=_warr[_cm]))
    ts=clamp(100-abs(tilt)*8)
    score=clamp(rot*0.28+bal*0.18+neg*0.15+conv*0.18+focal*0.12+ts*0.09)
    return score,{"rule_of_thirds":round(rot,2),"balance":round(bal,2),"negative_space":round(neg,2),
        "convergence":round(conv,2),"focal_strength":round(focal,2),"tilt_deg":round(tilt,3),
        "tilt_score":round(ts,1),"saliency_cx":ctx["saliency_cx"],"saliency_cy":ctx["saliency_cy"]}

# ── creative analysis ──────────────────────────────────────────────────────

def analyze_creative(ctx, color_m, cine_m, comp_m):
    elements=[]; concerns=[]
    ld=ctx["light_direction"]; bokeh=ctx["bokeh_detected"]; lum=ctx["luminosity_type"]; scene=ctx["scene_type"]
    harmony=ctx["harmony"]; sat=ctx["sat_mean"]
    if ld in ("top-left","top-right","left","right"):
        elements.append({"signal":"directional_light","label":f"Directional light ({ld})","note":"Creates depth, dimension and mood."})
    elif ld=="frontal" and scene=="subject":
        concerns.append({"signal":"flat_lighting","label":"Flat/frontal lighting","note":"Reduces subject dimensionality. Move key light 30-45 off-axis."})
    if bokeh and scene=="subject":
        strength="strong" if ctx["bokeh_ratio"]>3.0 else "moderate"
        elements.append({"signal":"bokeh","label":f"Subject isolation — {strength} bokeh","note":"Separates subject from background cinematically."})
    if lum=="LOW-KEY" and scene=="subject":
        elements.append({"signal":"low_key_portrait","label":"Low-key cinematic exposure","note":"Intentionally dark — moody and dramatic."})
    if harmony in ("complementary","split-complementary") and sat>0.2:
        elements.append({"signal":"color_harmony","label":f"{harmony.title()} color palette","note":"Visually dynamic color relationships."})
    elif harmony=="analogous" and sat>0.18:
        elements.append({"signal":"color_harmony","label":"Analogous color palette","note":"Harmonious, cohesive color feel."})
    if color_m.get("creative_grade")=="possible":
        elements.append({"signal":"creative_grade","label":"Intentional color grade detected","note":"Cast appears stylistic, not a WB error."})
    diag=ctx["diag_lines"]; horiz=ctx["horiz_lines"]
    if diag>15:
        elements.append({"signal":"leading_lines","label":"Strong leading lines","note":"Diagonal lines create depth and guide the eye."})
    elif horiz>60:
        elements.append({"signal":"perspective_lines","label":"Perspective convergence","note":"Lines converging toward vanishing point — cinematic depth."})
    focal=float(cine_m.get("focal_strength",50))
    if focal>70:
        elements.append({"signal":"strong_focal_point","label":"Strong visual focal point","note":"Viewer attention is well concentrated."})
    elif focal<35:
        concerns.append({"signal":"weak_focal_point","label":"Weak focal point","note":"Attention is scattered. Add a clear subject or point of interest."})
    rot=float(comp_m.get("rule_of_thirds",50))
    if rot>70 and scene=="subject":
        elements.append({"signal":"rule_of_thirds","label":"Strong rule-of-thirds placement","note":"Subject on a compositional intersection."})
    e_score=min(len(elements)*12,40); c_pen=min(len(concerns)*8,25)
    cine_boost=5 if lum=="LOW-KEY" and bokeh else 0
    creative_score=clamp(50.+e_score-c_pen+cine_boost)
    return {"creative_score":creative_score,"elements_detected":elements,"concerns":concerns,
        "summary":{"positive_signals":len(elements),"concerns":len(concerns),"bokeh_detected":bokeh,
                   "light_direction":ld,"color_harmony":harmony,"luminosity_type":lum}}

# ── suggestions ────────────────────────────────────────────────────────────

def _color_fix_hint(cast_desc: str) -> str:
    """Returns a software-agnostic correction hint based on cast description."""
    cd = cast_desc.lower()
    if "magenta" in cd:
        return "In your color tools, reduce the magenta channel or add green in the midtones."
    elif "green" in cd:
        return "In your color tools, reduce green or add a touch of magenta in the midtones."
    elif "cyan" in cd or "teal" in cd:
        return "Raise the color temperature slider or reduce the blue/cyan channel."
    elif "cool" in cd or "blue" in cd:
        return "Raise the color temperature (warmer) to neutralize the blue cast."
    elif "warm" in cd or "orange" in cd or "yellow" in cd:
        return "Lower the color temperature (cooler) or reduce the orange channel in midtones."
    else:
        return "Use the white balance or color mixer tool in your editor to neutralize the cast."

def build_suggestions(exp_m, color_m, cine_m, comp_m, sharp_m, ctx, creative):
    s=[]; scene=ctx["scene_type"]; lum=ctx["luminosity_type"]
    state=exp_m.get("state","ok")
    hl_hard=float(exp_m.get("highlight_clip",0)); hl_soft=float(exp_m.get("highlight_soft",0))
    sh_hard=float(exp_m.get("shadow_clip",0));   sh_soft=float(exp_m.get("shadow_soft",0))
    if state=="overexposed":
        if hl_hard>0.02: msg="Highlights are heavily clipped. Lower exposure ~1-1.5 stops to recover texture."
        elif hl_soft>0.05: msg="Image looks washed out. Lower exposure ~0.5-1 stop to restore depth in highlights."
        else: msg="Slightly overexposed. Reduce exposure ~0.5 stop or pull highlights in grading."
        s.append({"category":"Exposure","priority":"high","message":msg})
    elif state=="underexposed":
        is_lk=any(e["signal"]=="low_key_portrait" for e in creative["elements_detected"])
        if not is_lk:
            if sh_hard>0.02: msg="Blacks are severely crushed. Lift exposure ~1-1.5 stops or add fill light."
            elif sh_soft>0.05: msg="Image is underexposed. Lift ~0.5-1 stop or raise shadows in grading."
            else: msg="Slightly underexposed. Lift exposure ~0.5 stop or raise midtones in post."
            s.append({"category":"Exposure","priority":"high","message":msg})
    eff=float(color_m.get("effective_cast",0)); temp=color_m.get("temperature","neutral")
    tint=color_m.get("tint","neutral"); cg=color_m.get("creative_grade","unlikely")
    vision_cast=(creative.get("vision") or {}).get("color_cast_detail","")
    if eff>=8.0:
        if cg=="possible":
            cast_desc=vision_cast if vision_cast else temp
            s.append({"category":"Color","priority":"low","message":f"A {cast_desc} cast is present — if this is your intended grade, ignore this. Otherwise correct white balance before applying any look."})
        else:
            if vision_cast:
                fix=_color_fix_hint(vision_cast)
                s.append({"category":"Color","priority":"high" if eff>=12 else "medium","message":f"{vision_cast.capitalize()} detected. {fix}"})
            else:
                parts=[]
                if temp=="cool/cyan": parts.append("cyan/teal")
                elif temp=="cool":    parts.append("cool/blue")
                elif temp=="warm":    parts.append("warm/yellow-orange")
                if tint=="green":     parts.append("green tint")
                elif tint=="magenta": parts.append("magenta tint")
                label=" + ".join(parts) if parts else "color"
                fix=_color_fix_hint(label)
                s.append({"category":"Color","priority":"high" if eff>=12 else "medium","message":f"Noticeable {label} cast detected. {fix}"})
    sh_state=sharp_m.get("state","sharp"); ls=float(sharp_m.get("laplacian_std",50))
    bokeh=sharp_m.get("bokeh_detected",False); src=sharp_m.get("source","global")
    if sh_state in ("very_soft","soft") and not bokeh:
        msg=("Image appears very soft or hazy. Check focus, lens cleanliness, or motion blur." if ls<8 else
             "Image looks slightly soft. Ensure subject is in focus and try a faster shutter speed.")
        s.append({"category":"Sharpness","priority":"medium","message":msg})
    elif sh_state in ("very_soft","soft") and bokeh and src=="face_roi":
        s.append({"category":"Sharpness","priority":"medium","message":"Subject face appears soft even with bokeh. Check focus accuracy on the eyes."})
    if ctx["light_direction"]=="frontal" and scene=="subject":
        s.append({"category":"Lighting","priority":"low","message":"Lighting appears flat/frontal. Moving the key light 30-45 off-axis adds depth and dimension."})
    if scene=="subject":
        sep=float(cine_m.get("subject_separation",50))
        if sep<35 and ctx["bokeh_ratio"]<1.8:
            s.append({"category":"Cinematography","priority":"medium","message":"Subject separation is low. Increase subject-background distance or use a longer focal length."})
    else:
        if float(cine_m.get("layer_complexity",50))<35:
            s.append({"category":"Cinematography","priority":"low","message":"Scene looks flat. Add foreground elements or find leading lines to create depth."})
    tilt=float(comp_m.get("tilt_deg",0))
    if abs(tilt)>=2.5:
        s.append({"category":"Composition","priority":"medium","message":f"Lines tilted ~{abs(tilt):.1f}. Level the shot or correct rotation in post."})
    return s

# ── endpoint ───────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image    = cv2.imdecode(np.frombuffer(contents,np.uint8), cv2.IMREAD_COLOR)
    if image is None: return {"ok":False,"error":"Invalid image"}
    ctx        = build_scene_context(image)
    scene_type = ctx["scene_type"]
    exp_s,exp_m   = analyze_exposure(image,ctx)
    con_s,con_m   = analyze_contrast(image,ctx)
    col_s,col_m   = analyze_color_balance(image,ctx)
    noi_s,noi_m   = analyze_noise(image,ctx)
    sha_s,sha_m   = analyze_sharpness(image,ctx)
    cin_s,cin_m   = analyze_cinematography(image,ctx)
    comp_s,comp_m = analyze_composition(image,ctx)
    skin_s=skin_m=None
    if scene_type=="subject": skin_s,skin_m=analyze_skin(image,ctx)
    creative = analyze_creative(ctx,col_m,cin_m,comp_m)
    # Build known_issues and intentional_choices for Vision prompt
    _known_issues = []
    _intentional  = []
    lum_type = ctx.get("luminosity_type","")
    # Exposure flagging:
    # LOW-KEY: underexposed is ok/intentional, overexposed is still a problem
    # HIGH-KEY: overexposed is ok/intentional ONLY if highlights aren't clipped
    _hl_soft = float(exp_m.get("highlight_soft", 0))
    _high_key_truly_ok = lum_type == "HIGH-KEY" and _hl_soft < 0.03
    if exp_m.get("state") == "overexposed" and not _high_key_truly_ok:
        _known_issues.append("exposure overexposed")
    elif exp_m.get("state") == "underexposed" and lum_type != "LOW-KEY":
        _known_issues.append("exposure underexposed")
    if lum_type == "LOW-KEY" and exp_m.get("state") in ("ok","underexposed"):
        _intentional.append("low-key exposure is intentional — dark tones are a creative choice, not an error")
    elif _high_key_truly_ok and exp_m.get("state") == "overexposed":
        _intentional.append("high-key exposure is intentional — bright tones are a creative choice")
    if col_m.get("effective_cast",0) >= 6: _known_issues.append(f"color cast ({col_m.get('temperature','unknown')})")
    if sha_m.get("state") in ("very_soft","soft"): _known_issues.append("sharpness soft")
    elif sha_m.get("state") == "moderate" and scene_type == "ambient": _known_issues.append("sharpness moderate")
    # Pass bokeh as intentional if strong
    if ctx.get("bokeh_ratio",0) > 1.8:
        _intentional.append(f"background blur/bokeh is intentional subject isolation (ratio={ctx['bokeh_ratio']:.1f})")
    opencv_data = {**{k:v for k,v in ctx.items() if k!="_img"}, "tonal": ctx["tonal"],
                   "_known_issues": _known_issues, "_intentional": _intentional}
    vision   = await analyze_vision(image, opencv_data)
    # Pass issue count into creative so merge can apply caps
    creative["_known_issues_count"] = len(_known_issues)
    creative = merge_vision_into_creative(creative, vision, scene_type)
    cre_s = creative["creative_score"]
    suggestions = build_suggestions(exp_m,col_m,cin_m,comp_m,sha_m,ctx,creative)
    if scene_type=="subject":
        tech_score = exp_s*0.20+con_s*0.08+col_s*0.10+(skin_s or 50)*0.06+noi_s*0.04+sha_s*0.07+cin_s*0.10+comp_s*0.08
        cinematic_score = int(tech_score*0.50 + cre_s*0.50)
    else:
        tech_score = exp_s*0.22+con_s*0.14+col_s*0.18+noi_s*0.06+sha_s*0.08+cin_s*0.16+comp_s*0.11
        cinematic_score = int(tech_score*0.70 + cre_s*0.30)
    # Hard ceiling based on confirmed technical problems
    # Scenes with serious issues cannot score as high as clean shots
    n_issues = len(_known_issues)
    _issues_set = set(_known_issues)
    # Extra penalty if both primary perceptual problems are present together
    _has_exp   = any("exposure" in i for i in _issues_set)
    _has_color = any("color" in i for i in _issues_set)
    _has_sharp = any("sharpness" in i for i in _issues_set)
    _double_primary = _has_exp and _has_color  # worst combo: wrong exposure + wrong color
    _triple = _has_exp and _has_color and _has_sharp  # all three primary issues
    if _triple:
        cinematic_score = min(cinematic_score, 38)
    elif _double_primary:
        cinematic_score = min(cinematic_score, 45)
    elif n_issues >= 3:
        cinematic_score = min(cinematic_score, 45)
    elif n_issues == 2:
        cinematic_score = min(cinematic_score, 60)
    elif n_issues == 1 and scene_type == "ambient":
        cinematic_score = min(cinematic_score, 70)
    # Small boost for strong subject scenes with no issues at all
    elif n_issues == 0 and scene_type == "subject" and cre_s >= 70:
        cinematic_score = min(100, cinematic_score + 4)
    breakdown={"exposure":round(exp_s,1),"contrast":round(con_s,1),"color":round(col_s,1)}
    if scene_type=="subject" and skin_s is not None: breakdown["skin"]=round(skin_s,1)
    breakdown.update({"noise":round(noi_s,1),"sharpness":round(sha_s,1),
                      "cinematography":round(cin_s,1),"composition":round(comp_s,1),"creative":round(cre_s,1)})
    ctx_out = {k:v for k,v in ctx.items() if k!="_img"}
    return {"ok":True,"score":cinematic_score,"scene_type":scene_type,"breakdown":breakdown,
        "metrics":{"scene":ctx_out,"exposure":exp_m,"contrast":con_m,"color":col_m,
                   "noise":noi_m,"sharpness":sha_m,"cinematography":cin_m,"composition":comp_m,
                   **( {"skin":skin_m} if skin_m else {})},
        "creative":creative,"suggestions":suggestions,
        "_debug":{"known_issues":_known_issues,"n_issues":len(_known_issues),
                  "effective_cast":round(col_m.get("effective_cast",0),2),
                  "sha_state":sha_m.get("state"),"exp_state":exp_m.get("state"),
                  "lum_type":lum_type}}
