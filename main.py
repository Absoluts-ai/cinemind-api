from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

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
    # utile per verificare a colpo d’occhio quale versione è deployata
    return {"service": "cinemind-api-v4-suggestions"}

@app.get("/health")
def health():
    return {"ok": True, "service": "cinemind-api-v4-suggestions"}


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
# Exposure Analysis (Rec709-like)
# =============================

def analyze_exposure(image_bgr):
    """
    Exposure in Rec.709 preview should be judged by:
    - highlight clipping (pixels near 255)
    - shadow clipping (pixels near 0)
    - percentiles (p50, p95, p99) rather than only mean
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    mean = float(np.mean(gray))
    p5  = float(np.percentile(gray, 5))
    p50 = float(np.percentile(gray, 50))
    p95 = float(np.percentile(gray, 95))
    p99 = float(np.percentile(gray, 99))

    # clipping fractions
    highlight_clip = float(np.mean(gray >= 250))  # 0..1
    shadow_clip    = float(np.mean(gray <= 5))    # 0..1

    # exposure state (simple but robust for Rec709 previews)
    state = "ok"
    if highlight_clip > 0.015 or p95 > 235 or p99 > 248:
        state = "overexposed"
    elif shadow_clip > 0.015 or p50 < 55:
        state = "underexposed"

    # score: start at 100, penalize clipping heavily, then midtone drift
    score = 100.0

    # heavy penalties for clipping
    score -= min(60.0, highlight_clip * 3000.0)  # 1% -> -30
    score -= min(45.0, shadow_clip * 2500.0)     # 1% -> -25

    # midtone target for Rec709 previews (rough)
    # keep this light so we don't misclassify high-key shots
    target_mid = 120.0
    score -= min(20.0, abs(p50 - target_mid) * 0.15)

    score = clamp100(score)

    metrics = {
        "mean": round(mean, 3),
        "p5": round(p5, 3),
        "p50": round(p50, 3),
        "p95": round(p95, 3),
        "p99": round(p99, 3),
        "highlight_clip": round(highlight_clip, 6),
        "shadow_clip": round(shadow_clip, 6),
        "state": state
    }
    return score, metrics


# =============================
# Contrast Analysis (Rec709-like)
# =============================

def analyze_contrast(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))

    # keep your old idea but make it a bit safer
    # target std in many Rec709 previews: ~45-65 (very content dependent)
    target = 55.0
    score = 100.0 - abs(std - target) * 1.2
    return clamp100(score), float(std)


# =============================
# Color Balance / Cast Analysis (Lab-based)
# =============================

def analyze_color_balance(image_bgr):
    """
    Use Lab to detect cast:
    - a* negative = green, a* positive = magenta
    - b* negative = blue/cool, b* positive = yellow/warm
    """
    b, g, r = cv2.split(image_bgr)
    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV Lab: L in [0..255], a,b in [0..255] with 128 as neutral
    a = lab[:, :, 1] - 128.0
    bb = lab[:, :, 2] - 128.0

    a_mean = float(np.mean(a))
    b_lab_mean = float(np.mean(bb))

    cast_strength = float(np.sqrt(a_mean * a_mean + b_lab_mean * b_lab_mean))

    # score: penalize cast (cast_strength ~ 0-5 is near neutral; >12 obvious)
    score = 100.0 - cast_strength * 4.0
    score = clamp100(score)

    temperature = "neutral"
    if b_lab_mean > 6:
        temperature = "warm"
    elif b_lab_mean < -6:
        temperature = "cool"

    tint = "neutral"
    if a_mean > 6:
        tint = "magenta"
    elif a_mean < -6:
        tint = "green"

    metrics = {
        "r_mean": round(r_mean, 3),
        "g_mean": round(g_mean, 3),
        "b_mean": round(b_mean, 3),
        "lab_a_mean": round(a_mean, 3),
        "lab_b_mean": round(b_lab_mean, 3),
        "cast_strength": round(cast_strength, 3),
        "temperature": temperature,
        "tint": tint
    }
    return score, metrics


# =============================
# Skin Tone Detection (simple HSV mask)
# =============================

def analyze_skin(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    skin_count = int(np.count_nonzero(mask))
    ratio = float(skin_count / (total_pixels + 1e-6))

    if skin_count < 80 or ratio < 0.01:
        return 50.0, {"skin_detected": False, "skin_ratio": round(ratio, 6)}

    skin_pixels = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    b, g, r = cv2.split(skin_pixels)

    r_vals = r[mask > 0]
    g_vals = g[mask > 0]
    b_vals = b[mask > 0]

    r_mean = float(np.mean(r_vals))
    g_mean = float(np.mean(g_vals))
    b_mean = float(np.mean(b_vals))

    deviation = float(abs(r_mean - g_mean) + abs(r_mean - b_mean))
    score = 100.0 - deviation * 0.5
    score = clamp100(score)

    temperature = "neutral"
    if r_mean > b_mean + 15:
        temperature = "warm"
    elif b_mean > r_mean + 15:
        temperature = "cool"

    return score, {
        "skin_detected": True,
        "skin_ratio": round(ratio, 6),
        "r_mean": round(r_mean, 3),
        "g_mean": round(g_mean, 3),
        "b_mean": round(b_mean, 3),
        "temperature": temperature,
        "deviation": round(deviation, 3)
    }


# =============================
# Noise / Image Quality
# =============================

def analyze_noise(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_map = gray.astype(np.float32) - blur.astype(np.float32)

    noise_std = float(np.std(noise_map))
    mean_signal = float(np.mean(gray)) + 1e-6
    snr = float(mean_signal / (noise_std + 1e-6))

    noise_score = 100.0 - noise_std * 2.0
    noise_score = clamp100(noise_score)

    # optional texture proxy (keep but not required)
    texture = float(np.std(cv2.Laplacian(gray, cv2.CV_32F)))

    return noise_score, {
        "noise_std": round(noise_std, 6),
        "snr": round(snr, 6),
        "texture": round(texture, 6)
    }


# =============================
# Cinematography Analysis
# =============================

def analyze_cinematography(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    center, (x0, y0, cw, ch) = central_roi(gray, frac=0.45)
    outer_mask = outer_ring_mask(h, w, inner_frac=0.55)

    outer_pixels = gray[outer_mask > 0]
    center_std = float(np.std(center))
    outer_std = float(np.std(outer_pixels)) if outer_pixels.size > 0 else float(np.std(gray))

    sep_raw = center_std - outer_std
    subject_separation = clamp01((sep_raw + 15.0) / 40.0) * 100.0

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    center_lap = lap_abs[y0:y0 + ch, x0:x0 + cw]
    outer_lap = lap_abs[outer_mask > 0]

    center_sharp = float(np.mean(center_lap))
    outer_sharp = float(np.mean(outer_lap)) if outer_lap.size > 0 else float(np.mean(lap_abs))

    blur_ratio = (outer_sharp + 1e-6) / (center_sharp + 1e-6)
    background_blur = clamp01((1.10 - blur_ratio) / (1.10 - 0.35)) * 100.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_mean = float(np.mean(mag))

    p10 = float(np.percentile(center, 10))
    p90 = float(np.percentile(center, 90))
    spread = max(0.0, p90 - p10)

    modeling = clamp01((mag_mean - 6.0) / 18.0) * 0.45 + clamp01((spread - 30.0) / 80.0) * 0.55
    lighting_depth = clamp100(modeling * 100.0)

    thresh = float(np.percentile(gray, 95))
    bright_mask = (gray >= thresh).astype(np.uint8)
    ys, xs = np.where(bright_mask > 0)
    if xs.size < 50:
        directionality = 50.0
        bright_offset = 0.0
    else:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        dx = (cx - (w / 2.0)) / (w / 2.0)
        dy = (cy - (h / 2.0)) / (h / 2.0)
        bright_offset = float(np.sqrt(dx * dx + dy * dy))
        directionality = clamp100(35.0 + clamp01(bright_offset / 0.6) * 65.0)

    edges = cv2.Canny(gray, 60, 160)

    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[y0:y0 + ch, x0:x0 + cw] = 1

    mask_outer = outer_mask
    mask_mid = np.ones((h, w), dtype=np.uint8)
    mask_mid[(mask_center > 0)] = 0
    mask_mid[(mask_outer > 0)] = 0

    def edge_density(m):
        pix = int(np.count_nonzero(m))
        if pix <= 0:
            return 0.0
        return float(np.count_nonzero(edges[m > 0]) / pix)

    d_c = edge_density(mask_center)
    d_m = edge_density(mask_mid)
    d_o = edge_density(mask_outer)

    densities = np.array([d_c, d_m, d_o], dtype=np.float32)
    s = float(np.sum(densities)) + 1e-6
    p = densities / s
    entropy = float(-(p * np.log(p + 1e-6)).sum())
    layer_complexity = clamp100(clamp01(entropy / 1.05) * 100.0)

    cinematography_score = (
        subject_separation * 0.26 +
        lighting_depth     * 0.26 +
        background_blur    * 0.18 +
        layer_complexity   * 0.16 +
        directionality     * 0.14
    )
    cinematography_score = clamp100(cinematography_score)

    return cinematography_score, {
        "subject_separation": float(round(subject_separation, 2)),
        "background_blur": float(round(background_blur, 2)),
        "lighting_depth": float(round(lighting_depth, 2)),
        "layer_complexity": float(round(layer_complexity, 2)),
        "directionality": float(round(directionality, 2)),
        "bright_offset": float(round(bright_offset, 4)),
        "center_contrast_std": float(round(center_std, 3)),
        "outer_contrast_std": float(round(outer_std, 3)),
        "bg_sharpness": float(round(outer_sharp, 4)),
        "subject_sharpness": float(round(center_sharp, 4)),
    }


# =============================
# Composition Analysis (tilt fixed)
# =============================

def analyze_composition(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 60, 160)

    # edge energy centroid (simple "subject" proxy)
    ys, xs = np.where(edges > 0)
    if xs.size < 200:
        cx, cy = (w // 2), (h // 2)
    else:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

    # rule of thirds score: distance to nearest thirds intersection
    thirds = [
        (w / 3.0, h / 3.0), (2 * w / 3.0, h / 3.0),
        (w / 3.0, 2 * h / 3.0), (2 * w / 3.0, 2 * h / 3.0)
    ]
    dists = [np.hypot(cx - tx, cy - ty) for (tx, ty) in thirds]
    dmin = float(min(dists))
    # normalize by diagonal
    diag = float(np.hypot(w, h)) + 1e-6
    rule_of_thirds = clamp100((1.0 - (dmin / (0.55 * diag))) * 100.0)

    # balance: left vs right edge energy
    left_energy = float(np.mean(edges[:, :w // 2] > 0))
    right_energy = float(np.mean(edges[:, w // 2:] > 0))
    balance = clamp100((1.0 - min(1.0, abs(left_energy - right_energy) / 0.08)) * 100.0)

    # negative space: low edge density overall (more "clean" frame)
    edge_density = float(np.mean(edges > 0))
    negative_space = clamp100((1.0 - min(1.0, edge_density / 0.12)) * 100.0)

    # tilt detection via Hough lines, focus on near-horizontal segments
    tilt_deg = 0.0
    tilt_score = 100.0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80, minLineLength=80, maxLineGap=10)
    if lines is not None and len(lines) > 0:
        angles = []
        weights = []
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l
            dx = (x2 - x1)
            dy = (y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < 60:
                continue
            angle = np.degrees(np.arctan2(dy, dx))  # -180..180
            # map to [-90..90]
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180

            # keep near-horizontal lines for "horizon" tilt (|angle| <= 25 deg)
            if abs(angle) <= 25:
                angles.append(angle)
                weights.append(length)

        if len(angles) >= 2:
            angles = np.array(angles, dtype=np.float32)
            weights = np.array(weights, dtype=np.float32)
            # weighted median-ish: sort by angle then cumulative weights
            idx = np.argsort(angles)
            angles_s = angles[idx]
            weights_s = weights[idx]
            cw = np.cumsum(weights_s)
            mid = cw[-1] * 0.5
            k = int(np.searchsorted(cw, mid))
            tilt_deg = float(angles_s[min(k, len(angles_s) - 1)])
        elif len(angles) == 1:
            tilt_deg = float(angles[0])

        # score: 0deg => 100, 5deg => ~60, 10deg => ~20, cap
        tilt_score = clamp100(100.0 - abs(tilt_deg) * 8.0)

    # composition score (rough)
    comp_score = clamp100(
        rule_of_thirds * 0.35 +
        balance        * 0.25 +
        negative_space * 0.25 +
        tilt_score     * 0.15
    )

    metrics = {
        "subject_position": {"x": int(cx), "y": int(cy)},
        "edge_density": round(edge_density, 6),
        "tilt_deg": round(float(tilt_deg), 3),
        "tilt_score": round(float(tilt_score), 1),
        "rule_of_thirds": round(float(rule_of_thirds), 2),
        "balance": round(float(balance), 2),
        "negative_space": round(float(negative_space), 2),
    }
    return comp_score, metrics


# =============================
# Suggestions Engine
# =============================

def build_suggestions(exposure_metrics, color_metrics, cine_metrics, comp_metrics):
    s = []

    # Exposure suggestions (fixed: overexposed vs underexposed)
    state = exposure_metrics.get("state", "ok")
    hl_clip = exposure_metrics.get("highlight_clip", 0.0)
    sh_clip = exposure_metrics.get("shadow_clip", 0.0)
    p95 = exposure_metrics.get("p95", 0.0)

    if state == "overexposed":
        msg = "Highlights look clipped/too bright. Lower exposure ~0.5–1 stop (or reduce key light) to preserve detail."
        if hl_clip > 0.03 or p95 > 245:
            msg = "Highlights are heavily clipped. Lower exposure ~1 stop (or reduce key light) and protect whites to retain texture."
        s.append({"category": "Exposure", "priority": "high", "message": msg})

    elif state == "underexposed":
        msg = "Image looks underexposed. Increase exposure ~0.5–1 stop or add a soft key light to lift midtones without flattening contrast."
        if sh_clip > 0.03:
            msg = "Shadows are heavily crushed. Lift exposure or add fill to recover shadow detail while keeping contrast controlled."
        s.append({"category": "Exposure", "priority": "high", "message": msg})

    # Color cast suggestions (Lab-based)
    cast = float(color_metrics.get("cast_strength", 0.0))
    temp = color_metrics.get("temperature", "neutral")
    tint = color_metrics.get("tint", "neutral")

    if cast >= 10.0:
        parts = []
        if temp == "cool":
            parts.append("cool/blue")
        elif temp == "warm":
            parts.append("warm/yellow")
        if tint == "green":
            parts.append("green")
        elif tint == "magenta":
            parts.append("magenta")

        cast_label = " + ".join(parts) if parts else "color"
        s.append({
            "category": "Color",
            "priority": "high" if cast >= 14.0 else "medium",
            "message": f"Noticeable {cast_label} cast detected. Correct white balance (Temp/Tint) before doing any creative look."
        })

    # Cinematography: subject separation
    sep = float(cine_metrics.get("subject_separation", 50.0))
    if sep < 35.0:
        s.append({
            "category": "Cinematography",
            "priority": "medium",
            "message": "Subject separation looks limited. Increase subject–background distance, simplify background, or use a longer focal length."
        })

    # Composition: tilt only if real tilt
    tilt_deg = float(comp_metrics.get("tilt_deg", 0.0))
    if abs(tilt_deg) >= 2.0:
        s.append({
            "category": "Composition",
            "priority": "medium",
            "message": f"Horizon/lines appear tilted (~{abs(tilt_deg):.1f}°). Level the shot (or correct rotation in post) to improve perceived professionalism."
        })

    return s


# =============================
# Main Endpoint
# =============================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    # analyses
    exposure_score, exposure_metrics = analyze_exposure(image)
    contrast_score, contrast_value = analyze_contrast(image)
    color_score, color_metrics = analyze_color_balance(image)
    skin_score, skin_metrics = analyze_skin(image)
    noise_score, noise_metrics = analyze_noise(image)
    cine_score, cine_metrics = analyze_cinematography(image)
    comp_score, comp_metrics = analyze_composition(image)

    suggestions = build_suggestions(exposure_metrics, color_metrics, cine_metrics, comp_metrics)

    # Overall cinematic score (acquisition-focused)
    cinematic_score = int(
        (exposure_score * 0.22) +
        (contrast_score * 0.14) +
        (color_score * 0.16) +
        (skin_score * 0.14) +
        (noise_score * 0.10) +
        (cine_score * 0.14) +
        (comp_score * 0.10)
    )

    return {
        "ok": True,
        "score": cinematic_score,
        "breakdown": {
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1),
            "color": round(color_score, 1),
            "skin": round(skin_score, 1),
            "noise": round(noise_score, 1),
            "cinematography": round(cine_score, 1),
            "composition": round(comp_score, 1),
        },
        "metrics": {
            # exposure metrics now include state + clipping
            "exposure": exposure_metrics,
            "contrast": round(float(contrast_value), 6),
            "color": color_metrics,
            "skin": skin_metrics,
            "noise": noise_metrics,
            "cinematography": cine_metrics,
            "composition": comp_metrics
        },
        "suggestions": suggestions
    }
