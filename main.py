from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

# ============================================================
# CineMind API — Acquisition-focused analysis (Rec.709 preview)
# Advanced Composition (saliency-based) + fixed Tilt
# ============================================================

APP_VERSION = "cinemind-api-v3-compositionB-tiltfix"

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
    # This endpoint is ONLY to confirm you deployed the correct code.
    return {"ok": True, "service": "cinemind-api", "version": APP_VERSION}

@app.get("/health")
def health():
    return {"ok": True, "service": "cinemind-api", "version": APP_VERSION}


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

def normalize_map(m: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    m = m.astype(np.float32)
    mn = float(np.min(m))
    mx = float(np.max(m))
    if mx - mn < eps:
        return np.zeros_like(m, dtype=np.float32)
    return (m - mn) / (mx - mn + eps)


# =============================
# Exposure Analysis (Rec709-like)
# =============================

def analyze_exposure(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    # Rec.709 preview: target midtones around 115–135
    target = 125.0
    score = 100.0 - abs(mean - target) * 0.9
    return clamp100(score), mean


# =============================
# Contrast Analysis (Rec709-like)
# =============================

def analyze_contrast(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))

    # Cinematic “usable” contrast often around std ~ 45–65 (content-dependent)
    target = 55.0
    score = 100.0 - abs(std - target) * 1.2
    return clamp100(score), std


# =============================
# Color Balance Analysis
# =============================

def analyze_color_balance(image_bgr):
    b, g, r = cv2.split(image_bgr)
    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    rg = abs(r_mean - g_mean)
    rb = abs(r_mean - b_mean)
    gb = abs(g_mean - b_mean)

    cast_strength = float((rg + rb + gb) / 3.0)

    score = 100.0 - cast_strength
    score = clamp100(score)

    temperature = "neutral"
    if r_mean > b_mean + 10:
        temperature = "warm"
    elif b_mean > r_mean + 10:
        temperature = "cool"

    return score, {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "cast_strength": cast_strength,
        "temperature": temperature
    }


# =============================
# Skin Tone Detection (HSV mask, Rec709 preview)
# =============================

def analyze_skin(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    skin_count = int(np.count_nonzero(mask))
    ratio = float(skin_count / (total_pixels + 1e-6))

    if skin_count < 50 or ratio < 0.01:
        return 50.0, {"skin_detected": False, "skin_ratio": ratio}

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
        "skin_ratio": ratio,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "temperature": temperature,
        "deviation": deviation
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

    return noise_score, {
        "noise_std": noise_std,
        "snr": snr
    }


# =============================
# Cinematography Analysis (depth cues)
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
        lighting_depth       * 0.26 +
        background_blur      * 0.18 +
        layer_complexity     * 0.16 +
        directionality       * 0.14
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
# Composition Analysis — "B" (advanced, saliency-based)
#   - saliency centroid (main subject proxy)
#   - rule of thirds proximity
#   - visual balance (left/right + top/bottom)
#   - negative space (saliency concentration)
#   - tilt (robust Hough median angle)
# =============================

def _saliency_map(gray: np.ndarray) -> np.ndarray:
    """
    Lightweight saliency proxy (no ML):
    combine (1) gradient magnitude + (2) local contrast.
    returns normalized [0..1].
    """
    gray_f = gray.astype(np.float32)

    # gradient magnitude (structure)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = normalize_map(mag)

    # local contrast (difference from blurred)
    blur = cv2.GaussianBlur(gray_f, (0, 0), 7)
    lc = np.abs(gray_f - blur)
    lc = normalize_map(lc)

    sal = 0.55 * mag + 0.45 * lc
    sal = cv2.GaussianBlur(sal, (0, 0), 3)
    return normalize_map(sal)

def _weighted_centroid(wmap: np.ndarray):
    """
    Returns (cx, cy, mass) in pixel coords.
    """
    h, w = wmap.shape[:2]
    m = wmap.astype(np.float64)
    mass = float(m.sum())
    if mass <= 1e-9:
        return (w / 2.0, h / 2.0, 0.0)
    ys, xs = np.indices((h, w))
    cx = float((m * xs).sum() / mass)
    cy = float((m * ys).sum() / mass)
    return (cx, cy, mass)

def _tilt_from_hough(gray: np.ndarray):
    """
    Robust tilt estimate:
    - detect edges
    - Hough lines
    - take median angle relative to horizontal (degrees)
    - output abs tilt degrees in [0..90]
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 60, 180)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=70,
        minLineLength=max(40, int(min(gray.shape) * 0.12)),
        maxLineGap=15
    )

    if lines is None or len(lines) < 3:
        return 0.0, {"lines_used": 0}

    angles = []
    weights = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 30:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))  # -180..180
        # normalize to [-90..90] (treat 170° as -10° etc.)
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        angles.append(ang)
        weights.append(length)

    if len(angles) < 3:
        return 0.0, {"lines_used": len(angles)}

    angles = np.array(angles, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    # weighted median for robustness
    order = np.argsort(angles)
    a_sorted = angles[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted) / (np.sum(w_sorted) + 1e-6)
    idx = int(np.searchsorted(cdf, 0.5))
    med = float(a_sorted[min(idx, len(a_sorted) - 1)])

    tilt_deg = float(abs(med))
    tilt_deg = float(min(90.0, tilt_deg))
    return tilt_deg, {"lines_used": int(len(angles)), "median_angle": float(round(med, 3))}

def analyze_composition(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    sal = _saliency_map(gray)

    # focus on top salient areas (reduces background texture dominance)
    thr = float(np.percentile(sal, 85))
    sal2 = sal.copy()
    sal2[sal2 < thr] *= 0.25
    sal2 = normalize_map(sal2)

    cx, cy, mass = _weighted_centroid(sal2)

    # normalized position 0..1
    nx = float(cx / (w + 1e-6))
    ny = float(cy / (h + 1e-6))

    # --- Rule of thirds score ---
    thirds = [(1/3, 1/3), (2/3, 1/3), (1/3, 2/3), (2/3, 2/3)]
    dmin = 1e9
    for tx, ty in thirds:
        d = np.hypot(nx - tx, ny - ty)
        dmin = min(dmin, d)
    # typical max distance to nearest intersection ~ 0.47 (from corner-ish)
    rule_of_thirds = clamp100((1.0 - clamp01(dmin / 0.47)) * 100.0)

    # --- Balance score (saliency mass distribution) ---
    left_mass = float(sal2[:, : w // 2].sum())
    right_mass = float(sal2[:, w // 2 :].sum())
    top_mass = float(sal2[: h // 2, :].sum())
    bottom_mass = float(sal2[h // 2 :, :].sum())
    total = float(sal2.sum()) + 1e-6

    lr_imb = abs(left_mass - right_mass) / total
    tb_imb = abs(top_mass - bottom_mass) / total
    # lower imbalance = better balance
    balance = clamp100((1.0 - clamp01((lr_imb * 0.65 + tb_imb * 0.35) / 0.35)) * 100.0)

    # --- Negative space score ---
    # high if saliency is concentrated (subject pops) but not tiny
    sal_norm = sal2 / (sal2.sum() + 1e-6)
    entropy = float(-(sal_norm * np.log(sal_norm + 1e-9)).sum())
    # normalize entropy by log(N)
    ent_norm = float(entropy / (np.log(h * w + 1e-9)))
    # ent_norm high => spread everywhere (busy), low => concentrated
    negative_space = clamp100((1.0 - clamp01((ent_norm - 0.65) / 0.25)) * 100.0)

    # --- Edge density (scene busyness proxy) ---
    edges = cv2.Canny(gray, 70, 170)
    edge_density = float(np.count_nonzero(edges)) / float(h * w + 1e-6)

    # --- Tilt (fixed) ---
    tilt_deg, tilt_dbg = _tilt_from_hough(gray)
    # tilt score: 0° => 100, 10° => ~0 (hard penalty), cap at 15°
    tilt_score = clamp100((1.0 - clamp01(tilt_deg / 10.0)) * 100.0)

    # Overall composition score (no "tilt_score: 0" bug anymore)
    composition_score = clamp100(
        rule_of_thirds * 0.38 +
        balance        * 0.28 +
        negative_space * 0.22 +
        tilt_score     * 0.12
    )

    return composition_score, {
        "rule_of_thirds": float(round(rule_of_thirds, 2)),
        "balance": float(round(balance, 2)),
        "negative_space": float(round(negative_space, 2)),
        "tilt_degrees": float(round(tilt_deg, 3)),
        "tilt_score": float(round(tilt_score, 2)),
        "edge_density": float(round(edge_density, 6)),
        "subject_position": {"x": int(round(cx)), "y": int(round(cy))},
        "subject_position_norm": {"x": float(round(nx, 4)), "y": float(round(ny, 4))},
        "saliency_threshold_p85": float(round(thr, 4)),
        "tilt_debug": tilt_dbg
    }


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

    # Core analyses
    exposure_score, exposure_value = analyze_exposure(image)
    contrast_score, contrast_value = analyze_contrast(image)
    color_score, color_metrics = analyze_color_balance(image)
    skin_score, skin_metrics = analyze_skin(image)
    noise_score, noise_metrics = analyze_noise(image)
    cine_score, cine_metrics = analyze_cinematography(image)

    # NEW: Composition (B)
    comp_score, comp_metrics = analyze_composition(image)

    # Overall cinematic score (acquisition-focused)
    # (This judges "how cinematic the image is", not grading quality.)
    cinematic_score = int(
        (exposure_score * 0.18) +
        (contrast_score * 0.13) +
        (color_score * 0.12) +
        (skin_score * 0.14) +
        (noise_score * 0.11) +
        (cine_score * 0.18) +
        (comp_score * 0.14)
    )

    return {
        "ok": True,
        "version": APP_VERSION,
        "score": cinematic_score,
        "breakdown": {
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1),
            "color_balance": round(color_score, 1),
            "skin_tone": round(skin_score, 1),
            "noise": round(noise_score, 1),
            "cinematography": round(cine_score, 1),
            "composition": round(comp_score, 1),
        },
        "metrics": {
            "mean_luminance": round(float(exposure_value), 2),
            "contrast_std": round(float(contrast_value), 2),
            "color": color_metrics,
            "skin": skin_metrics,
            "noise": noise_metrics,
            "cinematography": cine_metrics,
            "composition": comp_metrics
        }
    }
