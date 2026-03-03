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

# =============================
# Health / Root
# =============================

@app.get("/")
def root():
    return {"service": "cinemind-api-v5"}

@app.get("/health")
def health():
    return {"ok": True, "service": "cinemind-api-v5"}


# =============================
# Utility
# =============================

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def clamp100(x: float) -> float:
    return float(max(0.0, min(100.0, x)))

def safe_resize_for_speed(img, max_dim=1280):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / m
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def central_roi(gray, frac=0.45):
    h, w = gray.shape[:2]
    ch = int(h * frac)
    cw = int(w * frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return gray[y0:y0 + ch, x0:x0 + cw], (x0, y0, cw, ch)

def outer_ring_mask(h, w, inner_frac=0.55):
    mask = np.ones((h, w), dtype=np.uint8)
    ch = int(h * inner_frac)
    cw = int(w * inner_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    mask[y0:y0 + ch, x0:x0 + cw] = 0
    return mask


# =============================
# Exposure Analysis
# =============================

def analyze_exposure(image_bgr):
    """
    Rec.709 exposure judged by:
    - Soft highlight clipping (>=230) and hard clipping (>=250)
    - Soft shadow clipping (<=30) and hard clipping (<=5)
    - Percentile distribution (p25, p50, p75)
    
    Catches 'soft overexposure' (e.g. hazy/washed scenes) that never
    reach 250 but are clearly too bright perceptually.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    mean  = float(np.mean(gray))
    p5    = float(np.percentile(gray, 5))
    p25   = float(np.percentile(gray, 25))
    p50   = float(np.percentile(gray, 50))
    p75   = float(np.percentile(gray, 75))
    p95   = float(np.percentile(gray, 95))
    p99   = float(np.percentile(gray, 99))

    hl_hard = float(np.mean(gray >= 250))   # hard clipping
    hl_soft = float(np.mean(gray >= 230))   # soft / near-clip
    sh_hard = float(np.mean(gray <= 5))     # crushed blacks
    sh_soft = float(np.mean(gray <= 30))    # heavy shadows

    # State detection — order matters (hard overexposure first)
    state = "ok"
    if hl_hard > 0.005 or hl_soft > 0.02 or p75 > 210 or p95 > 225:
        state = "overexposed"
    elif sh_hard > 0.01 or sh_soft > 0.03 or p25 < 30 or p50 < 60:
        state = "underexposed"

    # Score: start at 100, penalize deviations
    score = 100.0
    score -= min(50.0, hl_hard * 5000.0)   # hard clip: -50 at 1%
    score -= min(30.0, hl_soft * 600.0)    # soft clip: -30 at 5%
    score -= min(40.0, sh_hard * 4000.0)   # crushed blacks: -40 at 1%
    score -= min(20.0, sh_soft * 400.0)    # heavy shadows
    target_mid = 118.0
    score -= min(15.0, abs(p50 - target_mid) * 0.12)  # midtone offset penalty
    score = clamp100(score)

    return score, {
        "mean":           round(mean, 2),
        "p5":             round(p5, 2),
        "p25":            round(p25, 2),
        "p50":            round(p50, 2),
        "p75":            round(p75, 2),
        "p95":            round(p95, 2),
        "p99":            round(p99, 2),
        "highlight_clip": round(hl_hard, 6),
        "highlight_soft": round(hl_soft, 6),
        "shadow_clip":    round(sh_hard, 6),
        "shadow_soft":    round(sh_soft, 6),
        "state":          state,
    }


# =============================
# Contrast Analysis
# =============================

def analyze_contrast(image_bgr):
    """
    Contrast measured via usable tonal spread (p95-p5) rather than raw std,
    with a penalty when highlights are soft-clipped (image appears flat/washed).
    
    Target usable_spread ~200 for a well-exposed cinematic frame.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    std   = float(np.std(gray))
    p5    = float(np.percentile(gray, 5))
    p95   = float(np.percentile(gray, 95))
    usable_spread = p95 - p5

    hl_soft = float(np.mean(gray >= 230))

    spread_score = clamp100(100.0 - abs(usable_spread - 200.0) * 0.5)
    hl_penalty   = min(30.0, hl_soft * 400.0)
    score        = clamp100(spread_score - hl_penalty)

    return score, {
        "std":           round(std, 3),
        "usable_spread": round(usable_spread, 2),
        "p5":            round(p5, 2),
        "p95":           round(p95, 2),
    }


# =============================
# Color Balance / Cast Analysis
# =============================

def analyze_color_balance(image_bgr):
    """
    Cast detection via two complementary methods:
    1. CIELab a*/b* axes (standard)
    2. RGB channel imbalance — catches cyan casts that Lab can miss
       because cyan = G+B elevated vs R, which doesn't map cleanly to b*.

    Cast types reported: cool/cyan, cool, warm, neutral (temperature)
                         green, magenta, neutral (tint)
    """
    b_ch, g_ch, r_ch = cv2.split(image_bgr)
    r_mean = float(np.mean(r_ch))
    g_mean = float(np.mean(g_ch))
    b_mean = float(np.mean(b_ch))

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a_vals = lab[:, :, 1] - 128.0
    b_vals = lab[:, :, 2] - 128.0
    a_mean    = float(np.mean(a_vals))
    b_lab_mean = float(np.mean(b_vals))

    lab_cast      = float(np.sqrt(a_mean**2 + b_lab_mean**2))
    rgb_imbalance = float(np.std([r_mean, g_mean, b_mean]))

    # Cyan proxy: both G and B elevated relative to R
    cyan_proxy  = (g_mean + b_mean) / 2.0 - r_mean
    g_dominance = g_mean - r_mean
    b_dominance = b_mean - r_mean

    temperature = "neutral"
    if cyan_proxy > 8 and g_dominance > 10:
        temperature = "cool/cyan"
    elif b_dominance > 12 and g_dominance < 5:
        temperature = "cool"
    elif b_lab_mean < -5:
        temperature = "cool"
    elif b_lab_mean > 5 or (r_mean - b_mean > 12 and r_mean - g_mean > 8):
        temperature = "warm"

    tint = "neutral"
    if a_mean < -5:
        tint = "green"
    elif a_mean > 5:
        tint = "magenta"

    # Effective cast = worst of Lab-based or RGB-imbalance-based
    effective_cast = max(lab_cast, rgb_imbalance * 1.5)

    score = clamp100(100.0 - effective_cast * 3.5)

    return score, {
        "r_mean":        round(r_mean, 2),
        "g_mean":        round(g_mean, 2),
        "b_mean":        round(b_mean, 2),
        "lab_a_mean":    round(a_mean, 3),
        "lab_b_mean":    round(b_lab_mean, 3),
        "lab_cast":      round(lab_cast, 3),
        "rgb_imbalance": round(rgb_imbalance, 3),
        "effective_cast":round(effective_cast, 3),
        "cyan_proxy":    round(cyan_proxy, 3),
        "temperature":   temperature,
        "tint":          tint,
    }


# =============================
# Skin Tone Detection
# =============================

def analyze_skin(image_bgr):
    """
    HSV skin detection with:
    - Wider hue range (catches pale/overlit and darker skin)
    - Wrap-around mask for red hues (H near 180)
    - Lower saturation floor (handles desaturated/washed skin)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Primary range (orange-ish skin)
    lower1 = np.array([0,  15, 50], dtype=np.uint8)
    upper1 = np.array([25, 255, 255], dtype=np.uint8)
    # Wrap-around for deep red / brown skin tones
    lower2 = np.array([170, 15, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower1, upper1),
        cv2.inRange(hsv, lower2, upper2)
    )

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    skin_count   = int(np.count_nonzero(mask))
    ratio        = float(skin_count / (total_pixels + 1e-6))

    if skin_count < 80 or ratio < 0.008:
        return 50.0, {"skin_detected": False, "skin_ratio": round(ratio, 6)}

    b_ch, g_ch, r_ch = cv2.split(image_bgr)
    r_vals = r_ch[mask > 0].astype(np.float32)
    g_vals = g_ch[mask > 0].astype(np.float32)
    b_vals = b_ch[mask > 0].astype(np.float32)

    r_m = float(np.mean(r_vals))
    g_m = float(np.mean(g_vals))
    b_m = float(np.mean(b_vals))

    # Deviation from expected skin ratio (R > G > B)
    deviation = float(abs(r_m - g_m) + abs(r_m - b_m))
    score = clamp100(100.0 - deviation * 0.45)

    temperature = "neutral"
    if r_m > b_m + 18:
        temperature = "warm"
    elif b_m > r_m + 18:
        temperature = "cool"

    return score, {
        "skin_detected": True,
        "skin_ratio":    round(ratio, 6),
        "r_mean":        round(r_m, 2),
        "g_mean":        round(g_m, 2),
        "b_mean":        round(b_m, 2),
        "temperature":   temperature,
        "deviation":     round(deviation, 3),
    }


# =============================
# Noise Analysis
# =============================

def analyze_noise(image_bgr):
    """
    Gaussian residual noise (high-frequency) separate from sharpness.
    Low noise_std = clean sensor, high = visible grain/noise.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_map  = gray.astype(np.float32) - blur.astype(np.float32)
    noise_std  = float(np.std(noise_map))
    mean_signal= float(np.mean(gray)) + 1e-6
    snr        = float(mean_signal / (noise_std + 1e-6))

    # noise_std < 2: very clean; 5-10: acceptable; >15: visible grain
    score = clamp100(100.0 - noise_std * 2.5)

    return score, {
        "noise_std": round(noise_std, 4),
        "snr":       round(snr, 3),
    }


# =============================
# Sharpness Analysis (new, split from noise)
# =============================

def analyze_sharpness(image_bgr):
    """
    Laplacian std as sharpness proxy, log-scaled for perceptual linearity.
    Typical values:
      - In-focus iPhone shot, good light: lap_std ~50-150 -> score ~75-95
      - Soft / slight motion blur:        lap_std ~15-30  -> score ~55-65
      - Heavy blur / haze overlay:        lap_std ~2-8    -> score ~20-40
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    img  = safe_resize_for_speed(gray, max_dim=1280) if len(gray.shape) == 2 else gray
    lap  = cv2.Laplacian(img, cv2.CV_64F)
    lap_std = float(np.std(lap))

    # log scale: log(1+x)/log(1+80) * 100, capped at 100
    score = clamp100(math.log1p(lap_std) / math.log1p(80.0) * 100.0)

    state = "sharp"
    if lap_std < 10:
        state = "very_soft"
    elif lap_std < 20:
        state = "soft"
    elif lap_std < 40:
        state = "moderate"

    return score, {
        "laplacian_std": round(lap_std, 3),
        "state":         state,
    }


# =============================
# Cinematography Analysis
# =============================

def analyze_cinematography(image_bgr):
    img  = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    center, (x0, y0, cw, ch) = central_roi(gray, frac=0.45)
    outer_mask = outer_ring_mask(h, w, inner_frac=0.55)
    outer_pixels = gray[outer_mask > 0]

    center_std = float(np.std(center))
    outer_std  = float(np.std(outer_pixels)) if outer_pixels.size > 0 else float(np.std(gray))

    sep_raw = center_std - outer_std
    subject_separation = clamp01((sep_raw + 15.0) / 40.0) * 100.0

    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)
    center_lap  = lap_abs[y0:y0 + ch, x0:x0 + cw]
    outer_lap   = lap_abs[outer_mask > 0]
    center_sharp = float(np.mean(center_lap))
    outer_sharp  = float(np.mean(outer_lap)) if outer_lap.size > 0 else float(np.mean(lap_abs))

    blur_ratio       = (outer_sharp + 1e-6) / (center_sharp + 1e-6)
    background_blur  = clamp01((1.10 - blur_ratio) / (1.10 - 0.35)) * 100.0

    gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_mean = float(np.mean(mag))

    p10    = float(np.percentile(center, 10))
    p90    = float(np.percentile(center, 90))
    spread = max(0.0, p90 - p10)

    modeling      = clamp01((mag_mean - 6.0) / 18.0) * 0.45 + clamp01((spread - 30.0) / 80.0) * 0.55
    lighting_depth = clamp100(modeling * 100.0)

    thresh       = float(np.percentile(gray, 95))
    bright_mask  = (gray >= thresh).astype(np.uint8)
    ys, xs       = np.where(bright_mask > 0)
    bright_offset = 0.0
    if xs.size >= 50:
        cx = float(np.mean(xs)); cy = float(np.mean(ys))
        dx = (cx - (w / 2.0)) / (w / 2.0)
        dy = (cy - (h / 2.0)) / (h / 2.0)
        bright_offset = float(np.sqrt(dx * dx + dy * dy))
    directionality = clamp100(35.0 + clamp01(bright_offset / 0.6) * 65.0)

    edges = cv2.Canny(gray, 60, 160)
    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[y0:y0 + ch, x0:x0 + cw] = 1
    mask_outer  = outer_mask
    mask_mid    = np.ones((h, w), dtype=np.uint8)
    mask_mid[mask_center > 0] = 0
    mask_mid[mask_outer  > 0] = 0

    def edge_density(m):
        pix = int(np.count_nonzero(m))
        return 0.0 if pix <= 0 else float(np.count_nonzero(edges[m > 0]) / pix)

    p = np.array([edge_density(mask_center), edge_density(mask_mid), edge_density(mask_outer)], dtype=np.float32)
    p /= (float(np.sum(p)) + 1e-6)
    entropy = float(-(p * np.log(p + 1e-6)).sum())
    layer_complexity = clamp100(clamp01(entropy / 1.05) * 100.0)

    score = clamp100(
        subject_separation * 0.26 +
        lighting_depth     * 0.26 +
        background_blur    * 0.18 +
        layer_complexity   * 0.16 +
        directionality     * 0.14
    )

    return score, {
        "subject_separation":  round(subject_separation, 2),
        "background_blur":     round(background_blur, 2),
        "lighting_depth":      round(lighting_depth, 2),
        "layer_complexity":    round(layer_complexity, 2),
        "directionality":      round(directionality, 2),
        "bright_offset":       round(bright_offset, 4),
        "center_contrast_std": round(center_std, 3),
        "outer_contrast_std":  round(outer_std, 3),
        "bg_sharpness":        round(outer_sharp, 4),
        "subject_sharpness":   round(center_sharp, 4),
    }


# =============================
# Composition Analysis
# =============================

def analyze_composition(image_bgr):
    """
    Tilt fixed: use length-weighted mean of horizontal Hough lines
    instead of the previous sorted-array median which was unstable.
    """
    img  = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 60, 160)

    ys, xs = np.where(edges > 0)
    cx = int(np.mean(xs)) if xs.size >= 200 else w // 2
    cy = int(np.mean(ys)) if xs.size >= 200 else h // 2

    thirds = [
        (w / 3.0, h / 3.0), (2 * w / 3.0, h / 3.0),
        (w / 3.0, 2 * h / 3.0), (2 * w / 3.0, 2 * h / 3.0)
    ]
    dists = [np.hypot(cx - tx, cy - ty) for (tx, ty) in thirds]
    dmin  = float(min(dists))
    diag  = float(np.hypot(w, h)) + 1e-6
    rule_of_thirds = clamp100((1.0 - (dmin / (0.55 * diag))) * 100.0)

    left_energy  = float(np.mean(edges[:, :w // 2] > 0))
    right_energy = float(np.mean(edges[:, w // 2:] > 0))
    balance      = clamp100((1.0 - min(1.0, abs(left_energy - right_energy) / 0.08)) * 100.0)

    edge_density   = float(np.mean(edges > 0))
    negative_space = clamp100((1.0 - min(1.0, edge_density / 0.12)) * 100.0)

    # --- Tilt: length-weighted mean of horizontal Hough lines ---
    tilt_deg  = 0.0
    tilt_score = 100.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=80, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        angles_h = []
        weights  = []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            dx = float(x2 - x1); dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < 60:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            if angle > 90:  angle -= 180
            if angle < -90: angle += 180
            if abs(angle) <= 25:
                angles_h.append(angle)
                weights.append(length)

        if len(angles_h) >= 2:
            tilt_deg = float(np.average(np.array(angles_h), weights=np.array(weights)))
        elif len(angles_h) == 1:
            tilt_deg = float(angles_h[0])

        tilt_score = clamp100(100.0 - abs(tilt_deg) * 8.0)

    comp_score = clamp100(
        rule_of_thirds * 0.35 +
        balance        * 0.25 +
        negative_space * 0.25 +
        tilt_score     * 0.15
    )

    return comp_score, {
        "subject_position": {"x": cx, "y": cy},
        "edge_density":     round(edge_density, 6),
        "tilt_deg":         round(tilt_deg, 3),
        "tilt_score":       round(tilt_score, 1),
        "rule_of_thirds":   round(rule_of_thirds, 2),
        "balance":          round(balance, 2),
        "negative_space":   round(negative_space, 2),
    }


# =============================
# Suggestions Engine
# =============================

def build_suggestions(exposure_m, color_m, cine_m, comp_m, sharpness_m):
    s = []

    # --- Exposure ---
    state   = exposure_m.get("state", "ok")
    hl_hard = float(exposure_m.get("highlight_clip", 0.0))
    hl_soft = float(exposure_m.get("highlight_soft", 0.0))
    sh_hard = float(exposure_m.get("shadow_clip", 0.0))
    sh_soft = float(exposure_m.get("shadow_soft", 0.0))

    if state == "overexposed":
        if hl_hard > 0.02:
            msg = "Highlights are heavily clipped. Lower exposure ~1–1.5 stops or reduce key light to recover texture in whites."
        elif hl_soft > 0.05:
            msg = "Image looks washed out / over-bright. Lower exposure ~0.5–1 stop to restore depth and contrast in the highlights."
        else:
            msg = "Image is slightly overexposed. Reduce exposure ~0.5 stop or protect highlights with a gentle S-curve in grading."
        s.append({"category": "Exposure", "priority": "high", "message": msg})

    elif state == "underexposed":
        if sh_hard > 0.02:
            msg = "Blacks are severely crushed. Lift exposure ~1–1.5 stops or add fill light to recover shadow detail."
        elif sh_soft > 0.05:
            msg = "Image is underexposed with heavy shadows. Increase exposure ~0.5–1 stop or add soft fill to lift midtones."
        else:
            msg = "Image looks slightly underexposed. Increase exposure ~0.5 stop or raise midtones gently in post."
        s.append({"category": "Exposure", "priority": "high", "message": msg})

    # --- Color Cast ---
    effective_cast = float(color_m.get("effective_cast", 0.0))
    temp = color_m.get("temperature", "neutral")
    tint = color_m.get("tint", "neutral")

    if effective_cast >= 8.0:
        parts = []
        if temp == "cool/cyan": parts.append("cyan/teal")
        elif temp == "cool":    parts.append("cool/blue")
        elif temp == "warm":    parts.append("warm/yellow-orange")
        if tint == "green":     parts.append("green")
        elif tint == "magenta": parts.append("magenta")
        cast_label = " + ".join(parts) if parts else "color"
        priority   = "high" if effective_cast >= 13.0 else "medium"
        s.append({
            "category": "Color",
            "priority": priority,
            "message":  f"Noticeable {cast_label} cast detected. Correct white balance (Temp/Tint sliders) before applying any creative look.",
        })

    # --- Sharpness ---
    lap_std = float(sharpness_m.get("laplacian_std", 50.0))
    sharpness_state = sharpness_m.get("state", "sharp")
    if sharpness_state in ("very_soft", "soft"):
        if lap_std < 8:
            msg = "Image appears very soft or hazy. Check for lens fog, diffusion filter, or significant motion blur. Re-shoot if critical."
        else:
            msg = "Image looks slightly soft. Ensure subject is in focus; try a faster shutter speed to reduce motion blur."
        s.append({"category": "Sharpness", "priority": "medium", "message": msg})

    # --- Subject Separation ---
    sep = float(cine_m.get("subject_separation", 50.0))
    if sep < 35.0:
        s.append({
            "category": "Cinematography",
            "priority": "medium",
            "message":  "Subject separation is low. Increase subject–background distance, simplify the background, or use a longer focal length to compress and isolate the subject.",
        })

    # --- Tilt ---
    tilt_deg = float(comp_m.get("tilt_deg", 0.0))
    if abs(tilt_deg) >= 2.5:
        s.append({
            "category": "Composition",
            "priority": "medium",
            "message":  f"Horizon/lines appear tilted (~{abs(tilt_deg):.1f}°). Level the shot in-camera or correct rotation in post for a cleaner, more professional look.",
        })

    return s


# =============================
# Main Endpoint
# =============================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    exposure_score,   exposure_metrics   = analyze_exposure(image)
    contrast_score,   contrast_metrics   = analyze_contrast(image)
    color_score,      color_metrics      = analyze_color_balance(image)
    skin_score,       skin_metrics       = analyze_skin(image)
    noise_score,      noise_metrics      = analyze_noise(image)
    sharpness_score,  sharpness_metrics  = analyze_sharpness(image)
    cine_score,       cine_metrics       = analyze_cinematography(image)
    comp_score,       comp_metrics       = analyze_composition(image)

    suggestions = build_suggestions(
        exposure_metrics, color_metrics,
        cine_metrics, comp_metrics, sharpness_metrics
    )

    # Weighted cinematic score
    # Sharpness replaces the old noise-only metric in the breakdown
    cinematic_score = int(
        exposure_score   * 0.22 +
        contrast_score   * 0.12 +
        color_score      * 0.16 +
        skin_score       * 0.12 +
        noise_score      * 0.06 +
        sharpness_score  * 0.08 +
        cine_score       * 0.14 +
        comp_score       * 0.10
    )

    return {
        "ok":    True,
        "score": cinematic_score,
        "breakdown": {
            "exposure":    round(exposure_score, 1),
            "contrast":    round(contrast_score, 1),
            "color":       round(color_score, 1),
            "skin":        round(skin_score, 1),
            "noise":       round(noise_score, 1),
            "sharpness":   round(sharpness_score, 1),
            "cinematography": round(cine_score, 1),
            "composition": round(comp_score, 1),
        },
        "metrics": {
            "exposure":    exposure_metrics,
            "contrast":    contrast_metrics,
            "color":       color_metrics,
            "skin":        skin_metrics,
            "noise":       noise_metrics,
            "sharpness":   sharpness_metrics,
            "cinematography": cine_metrics,
            "composition": comp_metrics,
        },
        "suggestions": suggestions,
    }
