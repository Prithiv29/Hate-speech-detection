"""
Caste & migration hate-speech detector - FastAPI backend.

Run:   uvicorn app:app --reload
Open:  http://127.0.0.1:8000        (API docs: http://127.0.0.1:8000/docs)

Folder layout:
    app.py
    static/index.html
    indicbert2_trained/   <- unzip your Kaggle download here (model_state.pt, tokenizer files, info.json)

Optional environment variables:
    MODEL_DIR=path/to/model_folder     (default: ./indicbert2_trained)
    DEMO_MODE=1                        (fake predictions, no model needed - to preview the UI)
"""
import os, re, json, hashlib, threading, unicodedata
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE / "indicbert2_trained"))
DEMO = os.getenv("DEMO_MODE") == "1"
MAX_LEN = 128


# ---------- same light cleaning that was applied to the training data ----------
def clean_text(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = t.replace("\\n", " ")
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\S+", " ", t)
    t = re.sub(r"\b\d{1,2}:\d{2}(:\d{2})?\b", " ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return re.sub(r"\s+", " ", t.lower()).strip()


# ---------- the model wrapper ----------
class Predictor:
    def __init__(self):
        self.ready = False
        self.demo = DEMO
        self.device = "cpu"
        self.model_id = "demo"
        self._lock = threading.Lock()   # one prediction at a time (keeps CPU/GPU memory predictable)

    def load(self):
        if self.demo:
            self.ready = True
            return
        info_path, weights = MODEL_DIR / "info.json", MODEL_DIR / "model_state.pt"
        if not info_path.exists() or not weights.exists():
            raise FileNotFoundError(
                f"Model files not found in '{MODEL_DIR}'. Unzip your Kaggle download there so it contains "
                f"info.json, model_state.pt and the tokenizer files (or run with DEMO_MODE=1 to preview the UI).")
        # heavy imports only when a real model is needed
        import torch, torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        info = json.load(open(info_path))
        self.model_id = info["model_id"]
        self.torch = torch

        class EncoderMeanPool(nn.Module):          # must match the class used in training
            def __init__(self, config, dropout=0.2):
                super().__init__()
                self.enc = AutoModel.from_config(config)
                h = config.hidden_size
                self.drop = nn.Dropout(dropout)
                self.head = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 2))
            def forward(self, input_ids, attention_mask):
                hs = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                m = attention_mask.unsqueeze(-1).to(hs.dtype)
                return self.head(self.drop((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)))

        print(f"Loading tokenizer + architecture for {self.model_id} (needs internet the first time only) ...")
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        config = AutoConfig.from_pretrained(self.model_id)        # small file; weights come from our own checkpoint
        model = EncoderMeanPool(config)
        state = torch.load(weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        real_missing = [k for k in missing if "position_ids" not in k]
        if real_missing:
            raise RuntimeError(f"Checkpoint does not match the architecture. Missing keys: {real_missing[:5]} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.float().to(self.device).eval()
        self.ready = True
        print(f"Model ready on {self.device}.")

    def p_hate(self, cleaned: List[str]) -> List[float]:
        if self.demo:   # deterministic fake score so the UI can be previewed
            return [int(hashlib.md5(t.encode()).hexdigest()[:6], 16) / 0xFFFFFF for t in cleaned]
        torch = self.torch
        with self._lock, torch.inference_mode():
            b = self.tok(cleaned, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(b["input_ids"], b["attention_mask"])
            return logits.float().softmax(-1)[:, 1].cpu().tolist()


predictor = Predictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="Caste & Migration Hate Speech Detector (Tamil)", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
                   allow_methods=["*"], allow_headers=["*"])


# ---------- request / response schemas ----------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    threshold: float = Field(0.5, ge=0.05, le=0.95, description="P(hate) at or above this -> 'hate'")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)
    threshold: float = Field(0.5, ge=0.05, le=0.95)

class Prediction(BaseModel):
    label: str            # "hate" | "not_hate"
    is_hate: bool
    p_hate: float
    p_not_hate: float
    threshold: float
    cleaned_text: str
    demo: bool


def _run(texts: List[str], threshold: float) -> List[Prediction]:
    cleaned = [clean_text(t) for t in texts]
    if any(not c for c in cleaned):
        raise HTTPException(422, "A comment is empty after cleaning (only links, @mentions or timestamps?). Please enter some text.")
    out = []
    for c, p in zip(cleaned, predictor.p_hate(cleaned)):
        out.append(Prediction(label="hate" if p >= threshold else "not_hate", is_hate=p >= threshold,
                              p_hate=round(p, 4), p_not_hate=round(1 - p, 4), threshold=threshold,
                              cleaned_text=c, demo=predictor.demo))
    return out


# ---------- API ----------
@app.get("/api/health")
def health():
    return {"status": "ok" if predictor.ready else "loading", "model_loaded": predictor.ready,
            "model_id": predictor.model_id, "device": predictor.device, "demo": predictor.demo}

@app.post("/api/predict", response_model=Prediction)
def predict(req: PredictRequest):
    return _run([req.text], req.threshold)[0]

@app.post("/api/predict_batch", response_model=List[Prediction])
def predict_batch(req: BatchRequest):
    return _run(req.texts, req.threshold)


# ---------- frontend ----------
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE / "static" / "index.html")
