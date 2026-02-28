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


def analyze_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)

    # normalize to 0-100 score
    score = 100 - abs(mean - 127) * 0.8
    score = max(0, min(100, score))

    return score, mean


def analyze_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)

    score = min(100, std * 1.5)

    return score, std


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"ok": False, "error": "Invalid image"}

    exposure_score, exposure_value = analyze_exposure(image)
    contrast_score, contrast_value = analyze_contrast(image)

    cinematic_score = int((exposure_score * 0.6) + (contrast_score * 0.4))

    return {
        "ok": True,
        "score": cinematic_score,
        "breakdown": {
            "exposure": round(exposure_score, 1),
            "contrast": round(contrast_score, 1)
        },
        "metrics": {
            "mean_luminance": round(float(exposure_value), 2),
            "contrast_std": round(float(contrast_value), 2)
        }
    }
