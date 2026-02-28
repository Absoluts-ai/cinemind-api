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

@app.get("/health")
def health():
    return {"ok": True, "service": "cinemind-api"}


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
    # outer ring = everything outside a central rectangle
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

    # Cinematic “usable” contrast often around std ~ 45–65 (depends on content)
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

    # small cast is normal; penalize stronger cast
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
# Skin Tone Detection (simple HSV mask)
# =============================

def analyze_skin(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # broad skin range; works ok on Rec709 previews
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    skin_count = int(np.count_nonzero(mask))
    ratio = float(skin_count / (total_pixels + 1e-6))

    # if almost no skin, return neutral score and metrics
    if skin_count < 50 or ratio < 0.01:
        return 50.0, {
            "skin_detected": False,
            "skin_ratio": ratio
        }

    skin_pixels = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
    b, g, r = cv2.split(skin_pixels)

    r_vals = r[mask > 0]
    g_vals = g[mask > 0]
    b_vals = b[mask > 0]

    r_mean = float(np.mean(r_vals))
    g_mean = float(np.mean(g_vals))
    b_mean = float(np.mean(b_vals))

    # deviation proxy: how far channels diverge (very rough)
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

    # score: lower noise_std is better
    noise_score = 100.0 - noise_std * 2.0
    noise_score = clamp100(noise_score)

    return noise_score, {
        "noise_std": noise_std,
        "snr": snr
    }


# =============================
# Cinematography Analysis (includes "depth" cues)
# =============================

def analyze_cinematography(image_bgr):
    """
    Produces a cinematography score based on:
    - subject separation (center vs edges contrast)
    - background control / blur (edges sharpness vs center)
    - lighting depth (luminance gradients / modeling)
    - directionality (bright centroid offset)
    - layer complexity (edge distribution across rings)
    """
    img = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 1) Subject separation (center vs outer)
    center, (x0, y0, cw, ch) = central_roi(gray, frac=0.45)
    outer_mask = outer_ring_mask(h, w, inner_frac=0.55)

    outer_pixels = gray[outer_mask > 0]
    center_std = float(np.std(center))
    outer_std = float(np.std(outer_pixels)) if outer_pixels.size > 0 else float(np.std(gray))

    # If center has more contrast than edges => likely subject separation
    sep_raw = center_std - outer_std
    # Map roughly: -15..+25 => 0..1
    subject_separation = clamp01((sep_raw + 15.0) / 40.0) * 100.0

    # 2) Background blur estimation (Laplacian variance)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    center_lap = lap_abs[y0:y0 + ch, x0:x0 + cw]
    outer_lap = lap_abs[outer_mask > 0]

    center_sharp = float(np.mean(center_lap))
    outer_sharp = float(np.mean(outer_lap)) if outer_lap.size > 0 else float(np.mean(lap_abs))

    # If outer is softer than center => controlled background
    blur_ratio = (outer_sharp + 1e-6) / (center_sharp + 1e-6)
    # blur_ratio < 1 is good; map 0.35..1.10 => 1..0
    background_blur = clamp01((1.10 - blur_ratio) / (1.10 - 0.35)) * 100.0

    # 3) Lighting depth (modeling via gradients + tonal spread)
    # Sobel gradient magnitude (structure / modeling)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_mean = float(np.mean(mag))

    # tonal spread in center ROI (avoid pure noise, correlated with sculpting)
    p10 = float(np.percentile(center, 10))
    p90 = float(np.percentile(center, 90))
    spread = max(0.0, p90 - p10)

    # Combine: gradients + spread
    # mag_mean typical ~ 5..25; spread ~ 20..120
    modeling = clamp01((mag_mean - 6.0) / 18.0) * 0.45 + clamp01((spread - 30.0) / 80.0) * 0.55
    lighting_depth = clamp100(modeling * 100.0)

    # 4) Directionality (bright centroid offset)
    # Take top 5% brightest pixels, compute centroid distance from center
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
        bright_offset = float(np.sqrt(dx * dx + dy * dy))  # 0..~1.4
        # Too centered can mean flat; some offset suggests directional light.
        # Map 0..0.6 -> 35..100, cap beyond.
        directionality = clamp100(35.0 + clamp01(bright_offset / 0.6) * 65.0)

    # 5) Layer complexity (edge distribution across rings)
    edges = cv2.Canny(gray, 60, 160)
    # Split into 3 radial-ish zones (center, mid, outer) using masks
    mask_center = np.zeros((h, w), dtype=np.uint8)
    mask_center[y0:y0 + ch, x0:x0 + cw] = 1

    mask_outer = outer_mask
    mask_mid = np.ones((h, w), dtype=np.uint8)
    # mid = everything excluding center and outer
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

    # If edges are not all in one zone => more layering.
    densities = np.array([d_c, d_m, d_o], dtype=np.float32)
    s = float(np.sum(densities)) + 1e-6
    p = densities / s
    entropy = float(-(p * np.log(p + 1e-6)).sum())  # 0..~1.1
    layer_complexity = clamp100(clamp01(entropy / 1.05) * 100.0)

    # Combine cinematography score
    # weights: separation + lighting + bg control + layering + directionality
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
# Main Endpoint
# =============================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    exposure_score, exposure_value = analyze_exposure(image)
    contrast_score, contrast_value = analyze_contrast(image)
    color_score, color_metrics = analyze_color_balance(image)
    skin_score, skin_metrics = analyze_skin(image)
    noise_score, noise_metrics = analyze_noise(image)
    cine_score, cine_metrics = analyze_cinematography(image)

    # Overall cinematic score (acquisition-focused)
    cinematic_score = int(
        (exposure_score * 0.22) +
        (contrast_score * 0.16) +
        (color_score * 0.14) +
        (skin_score * 0.16) +
        (noise_score * 0.12) +
        (cine_score * 0.20)
    )

    return {
        "ok": True,
        "score": cinematic_score,
        "breakdown": {
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1),
            "color_balance": round(color_score, 1),
            "skin_tone": round(skin_score, 1),
            "noise": round(noise_score, 1),
            "cinematography": round(cine_score, 1),
        },
        "metrics": {
            "mean_luminance": round(float(exposure_value), 2),
            "contrast_std": round(float(contrast_value), 2),
            "color": color_metrics,
            "skin": skin_metrics,
            "noise": noise_metrics,
            "cinematography": cine_metrics
        }
    }
