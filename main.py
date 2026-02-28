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

@app.get("/health")
def health():
    return {"ok": True, "service": "cinemind-api"}


# =========================
# EXPOSURE
# =========================
def analyze_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)

    score = 100 - abs(mean - 127) * 0.8
    score = max(0, min(100, score))

    return score, mean


# =========================
# CONTRAST
# =========================
def analyze_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)

    score = min(100, std * 1.5)

    return score, std


# =========================
# COLOR BALANCE
# =========================
def analyze_color_balance(image):
    b, g, r = cv2.split(image)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    cast_strength = np.std([r_mean, g_mean, b_mean])

    if r_mean > b_mean + 5:
        temperature = "warm"
    elif b_mean > r_mean + 5:
        temperature = "cool"
    else:
        temperature = "neutral"

    score = 100 - cast_strength * 2
    score = max(0, min(100, score))

    return score, {
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "cast_strength": float(cast_strength),
        "temperature": temperature
    }


# =========================
# SKIN TONE
# =========================
def analyze_skin_tone(image):

    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    mask = cv2.inRange(img_ycrcb, lower, upper)

    skin_pixels = cv2.bitwise_and(image, image, mask=mask)

    skin_count = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]

    skin_ratio = skin_count / total_pixels if total_pixels > 0 else 0

    if skin_count == 0:
        return 0, {
            "skin_detected": False,
            "skin_ratio": 0
        }

    b, g, r = cv2.split(skin_pixels)

    r_mean = np.mean(r[mask > 0])
    g_mean = np.mean(g[mask > 0])
    b_mean = np.mean(b[mask > 0])

    deviation = abs(r_mean - g_mean) + abs(r_mean - b_mean)

    score = 100 - deviation * 0.5
    score = max(0, min(100, score))

    temperature = "warm" if r_mean > b_mean else "cool"

    return score, {
        "skin_detected": True,
        "skin_ratio": float(skin_ratio),
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "temperature": temperature,
        "deviation": float(deviation)
    }


# =========================
# NOISE
# =========================
def analyze_noise(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noise_std = np.std(gray)

    signal = np.mean(gray)
    snr = signal / (noise_std + 1e-6)

    texture = np.std(cv2.Laplacian(gray, cv2.CV_64F))

    score = min(100, snr * 3)

    return score, {
        "noise_std": float(noise_std),
        "snr": float(snr),
        "texture": float(texture)
    }


# =========================
# CINEMATOGRAPHY
# =========================
def analyze_cinematography(image):

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    center = gray[h//4:3*h//4, w//4:3*w//4]
    outer = gray.copy()

    center_std = np.std(center)
    outer_std = np.std(outer)

    subject_sharp = np.var(cv2.Laplacian(center, cv2.CV_64F))
    bg_sharp = np.var(cv2.Laplacian(outer, cv2.CV_64F))

    subject_sep = min(100, (subject_sharp / (bg_sharp + 1e-6)) * 20)
    blur_score = max(0, 100 - bg_sharp * 0.1)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    directionality = np.mean(np.abs(grad_x)) / (np.mean(np.abs(grad_y)) + 1e-6)
    directionality_score = min(100, directionality * 20)

    lighting_depth = min(100, abs(center_std - outer_std) * 2)

    return {
        "subject_separation": float(subject_sep),
        "background_blur": float(blur_score),
        "lighting_depth": float(lighting_depth),
        "directionality": float(directionality_score),
        "center_contrast_std": float(center_std),
        "outer_contrast_std": float(outer_std),
        "bg_sharpness": float(bg_sharp),
        "subject_sharpness": float(subject_sharp)
    }


# =========================
# COMPOSITION
# =========================
def analyze_composition(image):

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges > 0) / (h * w)

    # Center of mass (visual weight)
    moments = cv2.moments(edges)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx, cy = w // 2, h // 2

    # Rule of thirds points
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]

    def dist(x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)

    min_dist = min(
        dist(cx, cy, tx, ty)
        for tx in thirds_x
        for ty in thirds_y
    )

    max_dist = math.sqrt(w**2 + h**2)

    thirds_score = 100 - (min_dist / max_dist) * 100
    thirds_score = max(0, min(100, thirds_score))

    # Balance
    left_weight = np.sum(edges[:, :w//2] > 0)
    right_weight = np.sum(edges[:, w//2:] > 0)

    balance = 100 - abs(left_weight - right_weight) / (left_weight + right_weight + 1e-6) * 100

    # Negative space
    negative_space = 100 - edge_density * 500
    negative_space = max(0, min(100, negative_space))

    # Horizon tilt detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    tilt = 0
    if lines is not None:
        angles = []
        for line in lines[:10]:
            rho, theta = line[0]
            angle_deg = (theta * 180 / np.pi) - 90
            angles.append(angle_deg)
        tilt = np.mean(angles)

    tilt_score = 100 - abs(tilt) * 2
    tilt_score = max(0, min(100, tilt_score))

    composition_score = (
        thirds_score * 0.35 +
        balance * 0.25 +
        negative_space * 0.2 +
        tilt_score * 0.2
    )

    return composition_score, {
        "rule_of_thirds": float(thirds_score),
        "balance": float(balance),
        "negative_space": float(negative_space),
        "tilt": float(tilt),
        "tilt_score": float(tilt_score),
        "edge_density": float(edge_density),
        "subject_position": {
            "x": int(cx),
            "y": int(cy)
        }
    }


# =========================
# ANALYZE ENDPOINT
# =========================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    exposure_score, exposure_val = analyze_exposure(image)
    contrast_score, contrast_val = analyze_contrast(image)
    color_score, color_data = analyze_color_balance(image)
    skin_score, skin_data = analyze_skin_tone(image)
    noise_score, noise_data = analyze_noise(image)
    cinematography_data = analyze_cinematography(image)
    composition_score, composition_data = analyze_composition(image)

    cinematic_score = int(
        exposure_score * 0.2 +
        contrast_score * 0.2 +
        color_score * 0.15 +
        skin_score * 0.15 +
        noise_score * 0.15 +
        composition_score * 0.15
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
            "composition": round(composition_score, 1)
        },
        "metrics": {
            "mean_luminance": round(float(exposure_val), 2),
            "contrast_std": round(float(contrast_val), 2)
        },
        "color": color_data,
        "skin": skin_data,
        "noise": noise_data,
        "cinematography": cinematography_data,
        "composition": composition_data
    }
