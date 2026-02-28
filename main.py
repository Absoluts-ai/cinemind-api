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
# Exposure Analysis (Rec709)
# =============================

def analyze_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)

    # Rec709 cinematic target ~ 110–140 midtones
    target = 125
    score = 100 - abs(mean - target) * 0.9
    score = max(0, min(100, score))

    return score, mean


# =============================
# Contrast Analysis
# =============================

def analyze_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)

    # Cinematic contrast sweet spot
    target = 55
    score = 100 - abs(std - target) * 1.2
    score = max(0, min(100, score))

    return score, std


# =============================
# Color Balance Analysis
# =============================

def analyze_color_balance(image):

    b, g, r = cv2.split(image)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    rg_diff = abs(r_mean - g_mean)
    rb_diff = abs(r_mean - b_mean)
    gb_diff = abs(g_mean - b_mean)

    cast_strength = (rg_diff + rb_diff + gb_diff) / 3

    score = 100 - cast_strength
    score = max(0, min(100, score))

    temperature = "neutral"

    if r_mean > b_mean + 10:
        temperature = "warm"
    elif b_mean > r_mean + 10:
        temperature = "cool"

    return score, {
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "cast_strength": float(cast_strength),
        "temperature": temperature
    }


# =============================
# Skin Tone Detection
# =============================

def analyze_skin(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    skin_pixels = cv2.bitwise_and(image, image, mask=mask)

    total_pixels = image.shape[0] * image.shape[1]
    skin_count = np.count_nonzero(mask)

    ratio = skin_count / total_pixels

    skin_detected = ratio > 0.01

    if not skin_detected:
        return 50, {
            "skin_detected": False,
            "skin_ratio": float(ratio)
        }

    b, g, r = cv2.split(skin_pixels)

    r_vals = r[mask > 0]
    g_vals = g[mask > 0]
    b_vals = b[mask > 0]

    r_mean = np.mean(r_vals)
    g_mean = np.mean(g_vals)
    b_mean = np.mean(b_vals)

    deviation = abs((r_mean - g_mean)) + abs((r_mean - b_mean))

    score = 100 - deviation * 0.5
    score = max(0, min(100, score))

    temperature = "neutral"

    if r_mean > b_mean + 15:
        temperature = "warm"
    elif b_mean > r_mean + 15:
        temperature = "cool"

    return score, {
        "skin_detected": True,
        "skin_ratio": float(ratio),
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "temperature": temperature,
        "deviation": float(deviation)
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

    cinematic_score = int(
        (exposure_score * 0.30) +
        (contrast_score * 0.30) +
        (color_score * 0.20) +
        (skin_score * 0.20)
    )

    return {
        "ok": True,
        "score": cinematic_score,
        "breakdown": {
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1),
            "color_balance": round(color_score, 1),
            "skin_tone": round(skin_score, 1)
        },
        "metrics": {
            "mean_luminance": round(float(exposure_value), 2),
            "contrast_std": round(float(contrast_value), 2),
            "color": color_metrics,
            "skin": skin_metrics
        }
    }
