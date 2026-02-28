from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    size_bytes = len(data)

    return {
        "ok": True,
        "filename": file.filename,
        "size_bytes": size_bytes,
        "score": 72
    }
