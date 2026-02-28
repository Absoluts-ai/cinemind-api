from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

APP_VERSION = "cinemind-api-v4-suggestions"

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
    return {"service": APP_VERSION}

@app.get("/health")
def health():
    return {"ok": True, "service": APP_VERSION}


# =============================
# Utility
# =============================

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def clamp100(x):
    return max(0.0, min(100.0, float(x)))


# =============================
# Exposure
# =============================

def analyze_exposure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = float(np.mean(gray))

    target = 125.0
    score = 100 - abs(mean - target) * 0.9

    return clamp100(score), mean


# =============================
# Contrast
# =============================

def analyze_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))

    target = 55
    score = 100 - abs(std - target) * 1.2

    return clamp100(score), std


# =============================
# Color
# =============================

def analyze_color(img):
    b, g, r = cv2.split(img)

    r_mean = float(np.mean(r))
    g_mean = float(np.mean(g))
    b_mean = float(np.mean(b))

    cast = abs(r_mean - g_mean) + abs(r_mean - b_mean)

    score = clamp100(100 - cast * 0.5)

    return score, {
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean
    }


# =============================
# Skin
# =============================

def analyze_skin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 20, 70])
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    ratio = float(np.count_nonzero(mask) / (img.size / 3))

    if ratio < 0.01:
        return 50, {"skin_detected": False}

    skin_pixels = cv2.bitwise_and(img, img, mask=mask)
    b, g, r = cv2.split(skin_pixels)

    r_mean = float(np.mean(r[mask > 0]))
    g_mean = float(np.mean(g[mask > 0]))

    deviation = abs(r_mean - g_mean)

    score = clamp100(100 - deviation * 1.2)

    return score, {
        "skin_detected": True,
        "deviation": deviation
    }


# =============================
# Noise
# =============================

def analyze_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5),0)
    noise = gray.astype(np.float32) - blur.astype(np.float32)

    std = float(np.std(noise))

    score = clamp100(100 - std * 2)

    return score, {"noise_std": std}


# =============================
# Cinematography (simplified placeholder)
# =============================

def analyze_cinematography(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std = float(np.std(gray))

    score = clamp100(std)

    return score, {"contrast_proxy": std}


# =============================
# Composition (simplified placeholder)
# =============================

def analyze_composition(img):
    h, w = img.shape[:2]

    center_x = w / 2
    center_y = h / 2

    score = 80

    return score, {
        "center_x": center_x,
        "center_y": center_y,
        "tilt": 0
    }


# =============================
# Suggestions Engine
# =============================

def generate_suggestions(exposure, contrast, color, skin, noise, cine, comp):

    suggestions = []

    # Exposure
    if exposure < 60:
        suggestions.append({
            "category": "Exposure",
            "priority": "high",
            "message":
            "Your image appears underexposed. Increase exposure by about 0.5–1 stop or add a soft key light to recover midtones without flattening contrast."
        })

    if exposure > 85:
        suggestions.append({
            "category": "Exposure",
            "priority": "medium",
            "message":
            "Highlights may be approaching clipping. Consider reducing exposure slightly or controlling highlights with diffusion."
        })

    # Contrast
    if contrast < 55:
        suggestions.append({
            "category": "Contrast",
            "priority": "medium",
            "message":
            "The image contrast is relatively flat. Introducing directional lighting or increasing subject-background separation can improve depth."
        })

    # Color
    if color < 70:
        suggestions.append({
            "category": "Color",
            "priority": "medium",
            "message":
            "A noticeable color cast is present. Adjust white balance during shooting or use a neutral reference to improve color accuracy."
        })

    # Skin
    if skin < 65:
        suggestions.append({
            "category": "Skin Tone",
            "priority": "high",
            "message":
            "Skin tones deviate from natural balance. Adjust lighting color temperature or reduce mixed lighting sources."
        })

    # Noise
    if noise < 70:
        suggestions.append({
            "category": "Noise",
            "priority": "high",
            "message":
            "Visible noise is present. Increase light intensity rather than ISO to preserve dynamic range and detail."
        })

    # Cinematography
    if cine < 60:
        suggestions.append({
            "category": "Cinematography",
            "priority": "medium",
            "message":
            "Subject separation could be improved. Increase distance between subject and background or use a longer focal length."
        })

    # Composition
    if comp < 60:
        suggestions.append({
            "category": "Composition",
            "priority": "low",
            "message":
            "Consider repositioning the subject closer to rule-of-thirds points to enhance visual balance."
        })

    return suggestions


# =============================
# MAIN ENDPOINT
# =============================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"ok": False}

    exposure_score, exposure_val = analyze_exposure(img)
    contrast_score, contrast_val = analyze_contrast(img)
    color_score, color_metrics = analyze_color(img)
    skin_score, skin_metrics = analyze_skin(img)
    noise_score, noise_metrics = analyze_noise(img)
    cine_score, cine_metrics = analyze_cinematography(img)
    comp_score, comp_metrics = analyze_composition(img)

    suggestions = generate_suggestions(
        exposure_score,
        contrast_score,
        color_score,
        skin_score,
        noise_score,
        cine_score,
        comp_score
    )

    final_score = int(
        exposure_score * 0.2 +
        contrast_score * 0.15 +
        color_score * 0.15 +
        skin_score * 0.15 +
        noise_score * 0.15 +
        cine_score * 0.1 +
        comp_score * 0.1
    )

    return {
        "ok": True,
        "score": final_score,
        "breakdown": {
            "exposure": exposure_score,
            "contrast": contrast_score,
            "color": color_score,
            "skin": skin_score,
            "noise": noise_score,
            "cinematography": cine_score,
            "composition": comp_score
        },
        "metrics": {
            "exposure": exposure_val,
            "contrast": contrast_val,
            "color": color_metrics,
            "skin": skin_metrics,
            "noise": noise_metrics,
            "cinematography": cine_metrics,
            "composition": comp_metrics
        },
        "suggestions": suggestions
    }
