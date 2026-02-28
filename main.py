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

# ------------------------------------------------------------
# Meta
# ------------------------------------------------------------

SERVICE_NAME = "cinemind-api-v4-suggestions"


@app.get("/")
def root():
    # Useful for quick verification in Render + browser
    return {"service": SERVICE_NAME}


@app.get("/health")
def health():
    return {"ok": True, "service": SERVICE_NAME}


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Exposure Analysis v2 (Rec.709 preview)
# Fix: mean-luma alone is misleading (haze / lifted blacks / clipped highlights)
# ------------------------------------------------------------

def analyze_exposure_v2(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    mean = float(np.mean(gray))
    p1 = float(np.percentile(gray, 1))
    p5 = float(np.percentile(gray, 5))
    p50 = float(np.percentile(gray, 50))
    p95 = float(np.percentile(gray, 95))
    p99 = float(np.percentile(gray, 99))

    # clipping ratios (Rec709 8-bit)
    hi_clip = float(np.mean(gray >= 250))  # 0..1
    lo_clip = float(np.mean(gray <= 5))    # 0..1

    # dynamic range proxy
    dr = float(max(0.0, p99 - p1))

    # Determine state (simple and robust)
    # Overexposed if: significant highlight clipping OR very high p95/p99
    over = (hi_clip >= 0.01) or (p99 >= 248.0) or (p95 >= 235.0)
    # Underexposed if: significant shadow clipping OR very low p5/p1
    under = (lo_clip >= 0.02) or (p1 <= 3.0) or (p5 <= 12.0)

    if over and not under:
        state = "overexposed"
    elif under and not over:
        state = "underexposed"
    elif over and under:
        # both ends clipped => crushed + clipped / harsh grade or wrong preview
        state = "clipped_both"
    else:
        state = "ok"

    # Score model (penalize clipping heavily; mean only mildly)
    # Targets (Rec709-ish): p50 ~ 95–135 depending on scene; we keep it soft.
    target_mid = 115.0
    mid_penalty = abs(p50 - target_mid) * 0.25  # mild

    hi_penalty = (hi_clip * 900.0) + max(0.0, (p99 - 245.0) * 1.8) + max(0.0, (p95 - 230.0) * 0.9)
    lo_penalty = (lo_clip * 500.0) + max(0.0, (10.0 - p5) * 1.2) + max(0.0, (2.0 - p1) * 3.0)

    # If DR is extremely low, it often indicates haze / lifted blacks / flat preview
    # Not strictly exposure, but it affects "image density".
    dr_penalty = 0.0
    if dr < 60.0:
        dr_penalty = (60.0 - dr) * 0.35

    score = 100.0 - (mid_penalty + hi_penalty + lo_penalty + dr_penalty)
    score = clamp100(score)

    return score, {
        "mean_luma": mean,
        "p1": p1,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "highlight_clip_ratio": hi_clip,
        "shadow_clip_ratio": lo_clip,
        "dynamic_range_p99_p1": dr,
        "state": state,
    }


# ------------------------------------------------------------
# Contrast Analysis (Rec.709 preview)
# ------------------------------------------------------------

def analyze_contrast(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))

    # Cinematic “usable” contrast often around std ~ 45–65 (content-dependent)
    target = 55.0
    score = 100.0 - abs(std - target) * 1.2
    return clamp100(score), std


# ------------------------------------------------------------
# Color Balance Analysis v2 (temperature + tint, highlight-weighted)
# Fix: global RGB means miss casts that live in highlights / neutrals
# ------------------------------------------------------------

def _lab_stats(lab_img, mask=None):
    if mask is None:
        a = lab_img[:, :, 1].astype(np.float32)
        b = lab_img[:, :, 2].astype(np.float32)
    else:
        a = lab_img[:, :, 1][mask > 0].astype(np.float32)
        b = lab_img[:, :, 2][mask > 0].astype(np.float32)
        if a.size < 50 or b.size < 50:
            return None

    # OpenCV Lab: L in [0..255], a,b in [0..255] with 128 as "neutral"
    a_mean = float(np.mean(a) - 128.0)  # - => green, + => magenta
    b_mean = float(np.mean(b) - 128.0)  # - => blue (cool), + => yellow (warm)
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    return {
        "a_mean": a_mean,
        "b_mean": b_mean,
        "a_std": a_std,
        "b_std": b_std,
    }


def analyze_color_balance_v2(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1400)

    # For masks, use luminance from Rec709-like grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # Highlights = top 15% brightest pixels (more stable than a fixed threshold)
    thr_hi = float(np.percentile(gray, 85))
    hi_mask = (gray >= thr_hi).astype(np.uint8) * 255

    # Midtones = between 30th and 70th percentile
    lo_mid = float(np.percentile(gray, 30))
    hi_mid = float(np.percentile(gray, 70))
    mid_mask = ((gray >= lo_mid) & (gray <= hi_mid)).astype(np.uint8) * 255

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    global_stats = _lab_stats(lab, None)
    hi_stats = _lab_stats(lab, hi_mask)
    mid_stats = _lab_stats(lab, mid_mask)

    # Prefer highlights for WB cast detection (common in real shooting)
    use = hi_stats if hi_stats is not None else global_stats
    a = use["a_mean"]  # tint
    b = use["b_mean"]  # temp

    # Convert to interpretable labels
    # thresholds tuned for 8-bit Lab (rough but stable)
    def temp_label(b_mean):
        if b_mean <= -8:
            return "cool"
        if b_mean >= 8:
            return "warm"
        return "neutral"

    def tint_label(a_mean):
        if a_mean <= -6:
            return "green"
        if a_mean >= 6:
            return "magenta"
        return "neutral"

    temperature = temp_label(b)
    tint = tint_label(a)

    # Cast strength: combined magnitude (treat temp + tint)
    cast_strength = float(np.sqrt((a * a) + (b * b)))  # ~0..(large)
    # Score: allow some grade, but penalize stronger casts
    # If cast_strength <= ~6: basically neutral; beyond ~18 gets heavily penalized
    score = 100.0 - max(0.0, (cast_strength - 6.0) * 3.2)
    score = clamp100(score)

    # Also keep simple RGB means (useful to debug)
    bch, gch, rch = cv2.split(img)
    rgb_means = {
        "r_mean": float(np.mean(rch)),
        "g_mean": float(np.mean(gch)),
        "b_mean": float(np.mean(bch)),
    }

    metrics = {
        "rgb_means": rgb_means,
        "lab_global": global_stats,
        "lab_highlights": hi_stats,
        "lab_midtones": mid_stats,
        "temperature": temperature,
        "tint": tint,
        "cast_strength": cast_strength,
        "reference": "highlights" if hi_stats is not None else "global",
    }

    return score, metrics


# ------------------------------------------------------------
# Skin Tone Detection (simple HSV mask)
# Note: this is a heuristic; keep it as a component, not a judge of "grade"
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Noise / Image Quality
# ------------------------------------------------------------

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

    # texture proxy (how much high-frequency detail exists)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    texture = float(np.mean(np.abs(lap)))

    return noise_score, {
        "noise_std": noise_std,
        "snr": snr,
        "texture": texture
    }


# ------------------------------------------------------------
# Cinematography Analysis (depth cues)
# ------------------------------------------------------------

def analyze_cinematography(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1280)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 1) Subject separation (center vs outer)
    center, (x0, y0, cw, ch) = central_roi(gray, frac=0.45)
    outer_mask = outer_ring_mask(h, w, inner_frac=0.55)

    outer_pixels = gray[outer_mask > 0]
    center_std = float(np.std(center))
    outer_std = float(np.std(outer_pixels)) if outer_pixels.size > 0 else float(np.std(gray))

    sep_raw = center_std - outer_std
    subject_separation = clamp01((sep_raw + 15.0) / 40.0) * 100.0

    # 2) Background blur estimation (Laplacian variance)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.abs(lap)

    center_lap = lap_abs[y0:y0 + ch, x0:x0 + cw]
    outer_lap = lap_abs[outer_mask > 0]

    center_sharp = float(np.mean(center_lap))
    outer_sharp = float(np.mean(outer_lap)) if outer_lap.size > 0 else float(np.mean(lap_abs))

    blur_ratio = (outer_sharp + 1e-6) / (center_sharp + 1e-6)
    background_blur = clamp01((1.10 - blur_ratio) / (1.10 - 0.35)) * 100.0

    # 3) Lighting depth
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_mean = float(np.mean(mag))

    p10 = float(np.percentile(center, 10))
    p90 = float(np.percentile(center, 90))
    spread = max(0.0, p90 - p10)

    modeling = clamp01((mag_mean - 6.0) / 18.0) * 0.45 + clamp01((spread - 30.0) / 80.0) * 0.55
    lighting_depth = clamp100(modeling * 100.0)

    # 4) Directionality
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

    # 5) Layer complexity
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


# ------------------------------------------------------------
# Composition (rule of thirds + balance + negative space + FIXED TILT)
# Fix: tilt_score must NOT invert meaning; compute degrees from dominant near-horizontal lines
# ------------------------------------------------------------

def analyze_composition(image_bgr):
    img = safe_resize_for_speed(image_bgr, max_dim=1400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Edge map for geometry
    edges = cv2.Canny(gray, 60, 160)

    # --- Tilt / horizon estimation (Hough)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=int(min(w, h) * 0.20),
                            maxLineGap=20)

    tilt_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for l in lines[:300]:
            x1, y1, x2, y2 = l[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if abs(dx) < 1e-3:
                continue
            ang = np.degrees(np.arctan2(dy, dx))  # -180..180
            # bring angle to [-90..90]
            if ang > 90:
                ang -= 180
            if ang < -90:
                ang += 180
            # keep only near-horizontal lines (most indicative of tilt)
            if abs(ang) <= 20:
                length = np.hypot(dx, dy)
                angles.append((ang, length))

        if len(angles) >= 3:
            # weighted median-ish: sort by angle, weight by length
            angles_sorted = sorted(angles, key=lambda x: x[0])
            total_w = sum(a[1] for a in angles_sorted) + 1e-6
            cum = 0.0
            med = angles_sorted[0][0]
            for ang, wgt in angles_sorted:
                cum += wgt
                if cum >= total_w * 0.5:
                    med = ang
                    break
            tilt_deg = float(abs(med))
        else:
            tilt_deg = 0.0

    # Tilt score: 0deg => 100. 5deg => ~0-20 penalty. 10deg => strong penalty.
    tilt_score = clamp100(100.0 - tilt_deg * 12.0)

    # --- Subject proxy: use saliency-like center of edges (simple)
    ys, xs = np.where(edges > 0)
    if xs.size < 50:
        cx, cy = w // 2, h // 2
        edge_density = 0.0
    else:
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        edge_density = float(xs.size / (w * h))

    # Rule of thirds score: distance of (cx,cy) from nearest thirds intersection
    thirds_x = [w / 3.0, 2.0 * w / 3.0]
    thirds_y = [h / 3.0, 2.0 * h / 3.0]
    pts = [(tx, ty) for tx in thirds_x for ty in thirds_y]
    d = min(np.hypot(cx - tx, cy - ty) for tx, ty in pts)
    d_norm = d / (np.hypot(w, h) + 1e-6)
    rule_of_thirds = clamp100(100.0 - d_norm * 260.0)

    # Balance: compare left/right edge energy
    left = edges[:, :w // 2]
    right = edges[:, w // 2:]
    eL = float(np.count_nonzero(left))
    eR = float(np.count_nonzero(right))
    bal = 1.0 - (abs(eL - eR) / (eL + eR + 1e-6))
    balance = clamp100(bal * 100.0)

    # Negative space proxy: how much of frame is "low edge"
    # If edges sparse => more negative space (can be good)
    negative_space = clamp100((1.0 - clamp01(edge_density / 0.08)) * 100.0)

    # Final composition score (simple)
    comp_score = clamp100(
        rule_of_thirds * 0.40 +
        balance        * 0.25 +
        negative_space * 0.20 +
        tilt_score     * 0.15
    )

    return comp_score, {
        "rule_of_thirds": float(round(rule_of_thirds, 2)),
        "balance": float(round(balance, 2)),
        "negative_space": float(round(negative_space, 2)),
        "tilt_degrees": float(round(tilt_deg, 3)),
        "tilt_score": float(round(tilt_score, 2)),
        "edge_density": float(round(edge_density, 6)),
        "subject_position": {"x": int(cx), "y": int(cy)},
    }


# ------------------------------------------------------------
# Suggestions Engine (now driven by Exposure v2 + Color v2)
# ------------------------------------------------------------

def build_suggestions(exposure_metrics, color_metrics, cine_metrics, comp_metrics):
    suggestions = []

    # Exposure suggestions
    state = exposure_metrics.get("state", "ok")
    hi_clip = float(exposure_metrics.get("highlight_clip_ratio", 0.0))
    lo_clip = float(exposure_metrics.get("shadow_clip_ratio", 0.0))
    p95 = float(exposure_metrics.get("p95", 0.0))
    p5 = float(exposure_metrics.get("p5", 0.0))

    if state in ("overexposed", "clipped_both"):
        msg = "Highlights look pushed/clipped. Lower exposure ~0.5–1 stop and protect highlights (reduce ISO/exposure, add ND, or reposition key light)."
        if hi_clip >= 0.03 or p95 >= 240:
            msg = "Highlights are clipping. Reduce exposure ~1 stop and protect whites (ND filter, lower ISO, or reduce key intensity)."
        suggestions.append({"category": "Exposure", "priority": "high", "message": msg})

    if state in ("underexposed", "clipped_both"):
        msg = "Shadows are too low/crushed. Raise exposure ~0.5 stop or add fill to lift midtones while keeping contrast."
        if lo_clip >= 0.05 or p5 <= 10:
            msg = "Shadows are clipping/crushing. Increase exposure or add soft fill to recover shadow detail."
        suggestions.append({"category": "Exposure", "priority": "high" if state == "underexposed" else "medium", "message": msg})

    # Color temperature / tint suggestions
    temp = color_metrics.get("temperature", "neutral")
    tint = color_metrics.get("tint", "neutral")
    cast_strength = float(color_metrics.get("cast_strength", 0.0))

    if cast_strength >= 14.0:
        # Temperature
        if temp == "cool":
            suggestions.append({
                "category": "Color",
                "priority": "high",
                "message": "Strong cool/blue cast detected. Warm up white balance (increase temperature) and re-check neutrals in highlights."
            })
        elif temp == "warm":
            suggestions.append({
                "category": "Color",
                "priority": "high",
                "message": "Strong warm/yellow cast detected. Cool down white balance (decrease temperature) and re-check neutrals in highlights."
            })

        # Tint
        if tint == "green":
            suggestions.append({
                "category": "Color",
                "priority": "high",
                "message": "Green cast detected. Add a touch of magenta tint (or reduce green in highlights) to neutralize whites."
            })
        elif tint == "magenta":
            suggestions.append({
                "category": "Color",
                "priority": "high",
                "message": "Magenta cast detected. Add a touch of green tint to bring neutrals back to center."
            })

    # Cinematography / separation suggestion (soft)
    subj_sep = float(cine_metrics.get("subject_separation", 50.0))
    bg_blur = float(cine_metrics.get("background_blur", 50.0))
    if subj_sep < 35.0 and bg_blur < 55.0:
        suggestions.append({
            "category": "Cinematography",
            "priority": "medium",
            "message": "Subject separation looks limited. Increase subject–background distance, simplify background, or use a longer focal length."
        })

    # Composition / tilt suggestion
    tilt_deg = float(comp_metrics.get("tilt_degrees", 0.0))
    if tilt_deg >= 3.5:
        suggestions.append({
            "category": "Composition",
            "priority": "medium",
            "message": "Horizon/lines appear tilted. Level the shot (or correct rotation in post) to improve perceived professionalism."
        })

    return suggestions


# ------------------------------------------------------------
# Main Endpoint
# ------------------------------------------------------------

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    # Core analysis
    exposure_score, exposure_metrics = analyze_exposure_v2(image)
    contrast_score, contrast_value = analyze_contrast(image)
    color_score, color_metrics = analyze_color_balance_v2(image)
    skin_score, skin_metrics = analyze_skin(image)
    noise_score, noise_metrics = analyze_noise(image)
    cine_score, cine_metrics = analyze_cinematography(image)
    comp_score, comp_metrics = analyze_composition(image)

    # Overall score (acquisition-focused: "how cinematic is the image you captured")
    cinematic_score = int(
        (exposure_score * 0.22) +
        (contrast_score * 0.14) +
        (color_score * 0.16) +
        (skin_score * 0.12) +
        (noise_score * 0.12) +
        (cine_score * 0.16) +
        (comp_score * 0.08)
    )

    suggestions = build_suggestions(exposure_metrics, color_metrics, cine_metrics, comp_metrics)

    return {
        "ok": True,
        "score": cinematic_score,
        "breakdown": {
            "exposure": float(round(exposure_score, 2)),
            "contrast": float(round(contrast_score, 2)),
            "color": float(round(color_score, 2)),
            "skin": float(round(skin_score, 2)),
            "noise": float(round(noise_score, 2)),
            "cinematography": float(round(cine_score, 2)),
            "composition": float(round(comp_score, 2)),
        },
        "metrics": {
            "exposure": exposure_metrics,                # includes state + clipping + percentiles
            "contrast_std": float(round(contrast_value, 4)),
            "color": color_metrics,                      # includes temperature + tint + highlight-weighted Lab
            "skin": skin_metrics,
            "noise": noise_metrics,
            "cinematography": cine_metrics,
            "composition": comp_metrics,                 # includes tilt_degrees + tilt_score (fixed meaning)
        },
        "suggestions": suggestions
    }
