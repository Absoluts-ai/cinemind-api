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

    b_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    r_mean = np.mean(image[:,:,2])

    rg_diff = abs(r_mean - g_mean)
    rb_diff = abs(r_mean - b_mean)
    gb_diff = abs(g_mean - b_mean)

    cast_strength = (rg_diff + rb_diff + gb_diff) / 3

    score = max(0, 100 - cast_strength * 1.5)

    # temperature estimate
    if r_mean > b_mean + 5:
        temp = "warm"
    elif b_mean > r_mean + 5:
        temp = "cool"
    else:
        temp = "neutral"

    return score, {
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "cast_strength": float(cast_strength),
        "temperature": temp
    }


# =========================
# SKIN TONE ANALYSIS
# =========================

def analyze_skin_tone(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 30, 60])
    upper = np.array([25, 180, 255])

    mask = cv2.inRange(hsv, lower, upper)

    skin_pixels = cv2.bitwise_and(image, image, mask=mask)

    skin_count = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]

    skin_ratio = skin_count / total_pixels

    if skin_count < 50:
        return 50, {
            "skin_detected": False,
            "skin_ratio": float(skin_ratio)
        }

    r_mean = np.mean(skin_pixels[:,:,2][mask > 0])
    g_mean = np.mean(skin_pixels[:,:,1][mask > 0])
    b_mean = np.mean(skin_pixels[:,:,0][mask > 0])

    warmth = r_mean - b_mean

    deviation = abs(warmth - 25)

    score = max(0, 100 - deviation * 1.2)

    temp = "warm" if warmth > 20 else "neutral"

    return score, {
        "skin_detected": True,
        "skin_ratio": float(skin_ratio),
        "r_mean": float(r_mean),
        "g_mean": float(g_mean),
        "b_mean": float(b_mean),
        "temperature": temp,
        "deviation": float(deviation)
    }


# =========================
# NOISE / IMAGE QUALITY
# =========================

def analyze_noise(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth version
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # high frequency component
    noise_map = gray.astype(np.float32) - blur.astype(np.float32)

    noise_std = np.std(noise_map)

    mean_signal = np.mean(gray) + 1e-6

    snr = mean_signal / (noise_std + 1e-6)

    texture = np.std(gray)

    # score calculation
    noise_score = max(0, 100 - noise_std * 2.0)

    return noise_score, {
        "noise_std": float(noise_std),
        "snr": float(snr),
        "texture": float(texture)
    }


# =========================
# MAIN ENDPOINT
# =========================

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
    skin_score, skin_metrics = analyze_skin_tone(image)
    noise_score, noise_metrics = analyze_noise(image)

    cinematic_score = int(
        (exposure_score * 0.25) +
        (contrast_score * 0.20) +
        (color_score * 0.20) +
        (skin_score * 0.20) +
        (noise_score * 0.15)
    )

    return {
        "ok": True,
        "score": cinematic_score,
        "breakdown": {
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1),
            "color_balance": round(color_score, 1),
            "skin_tone": round(skin_score, 1),
            "noise": round(noise_score, 1)
        },
        "metrics": {
            "mean_luminance": round(float(exposure_value), 2),
            "contrast_std": round(float(contrast_value), 2),
            "color": color_metrics,
            "skin": skin_metrics,
            "noise": noise_metrics
        }
    }
