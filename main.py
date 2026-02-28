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


# -------------------------
# EXPOSURE ANALYSIS
# -------------------------
def analyze_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)

    score = 100 - abs(mean - 127) * 0.8
    score = max(0, min(100, score))

    return score, mean


# -------------------------
# CONTRAST ANALYSIS
# -------------------------
def analyze_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)

    score = min(100, std * 1.5)

    return score, std


# -------------------------
# COLOR BALANCE ANALYSIS
# -------------------------
def analyze_color_balance(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r_mean = np.mean(rgb[:, :, 0])
    g_mean = np.mean(rgb[:, :, 1])
    b_mean = np.mean(rgb[:, :, 2])

    r_diff = r_mean - g_mean
    b_diff = b_mean - g_mean

    cast_strength = abs(r_diff) + abs(b_diff)

    score = max(0, 100 - cast_strength * 0.5)

    if r_mean > b_mean:
        temperature = "warm"
    elif b_mean > r_mean:
        temperature = "cool"
    else:
        temperature = "neutral"

    color_data = {
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "cast_strength": float(cast_strength),
        "temperature": temperature
    }

    return score, color_data


# -------------------------
# SKIN TONE ANALYSIS
# -------------------------
def analyze_skin_tones(image):

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Skin detection range
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)

    mask = cv2.inRange(ycrcb, lower, upper)

    skin_pixels = cv2.bitwise_and(image, image, mask=mask)

    skin_count = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]

    skin_ratio = skin_count / total_pixels

    if skin_count == 0:
        return 50, {
            "skin_detected": False,
            "skin_ratio": 0
        }

    # Mean skin color
    skin_rgb = cv2.cvtColor(skin_pixels, cv2.COLOR_BGR2RGB)
    r = skin_rgb[:, :, 0][mask > 0]
    g = skin_rgb[:, :, 1][mask > 0]
    b = skin_rgb[:, :, 2][mask > 0]

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    # Ideal cinematic skin reference
    ideal_r = 180
    ideal_g = 140
    ideal_b = 120

    deviation = abs(r_mean - ideal_r) + abs(g_mean - ideal_g) + abs(b_mean - ideal_b)

    score = max(0, 100 - deviation * 0.3)

    # Temperature estimation
    if r_mean > b_mean:
        temp = "warm"
    elif b_mean > r_mean:
        temp = "cool"
    else:
        temp = "neutral"

    data = {
        "skin_detected": True,
        "skin_ratio": float(skin_ratio),
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "temperature": temp,
        "deviation": float(deviation)
    }

    return score, data


# -------------------------
# MAIN ANALYZE ENDPOINT
# -------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    exposure_score, exposure_value = analyze_exposure(image)
    contrast_score, contrast_value = analyze_contrast(image)
    color_score, color_data = analyze_color_balance(image)
    skin_score, skin_data = analyze_skin_tones(image)

    cinematic_score = int(
        (exposure_score * 0.3) +
        (contrast_score * 0.25) +
        (color_score * 0.25) +
        (skin_score * 0.2)
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
            "color": color_data,
            "skin": skin_data
        }
    }
