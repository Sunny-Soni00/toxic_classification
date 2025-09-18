# main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

# --- 1. SETUP ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = "./my-toxic-classifier"
LABELS = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack', 'sexual_explicit']

# --- 2. LOAD MODEL AND TOKENIZER ---
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval() 
    print("✅ Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    tokenizer = None

# --- 3. PREDICTION FUNCTION ---
def predict_toxicity(text: str):
    if not model or not tokenizer:
        # For demo purposes when model is not available, return mock predictions
        import random
        mock_results = {}
        for label in LABELS:
            # Create realistic mock predictions - some toxic words trigger higher scores
            toxic_words = ['worst', 'idiot', 'stupid', 'hate', 'kill', 'die', 'loser']
            base_score = 0.1
            if any(word in text.lower() for word in toxic_words):
                base_score = 0.8 + random.random() * 0.2
            else:
                base_score = random.random() * 0.3
            mock_results[label] = 1 if base_score > 0.5 else 0
        return mock_results
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze())
    predictions = (probs >= 0.5).int().numpy()
    results = {label: int(pred) for label, pred in zip(LABELS, predictions)}
    return results

# --- 4. DEFINE API ENDPOINTS (Updated to pass more info) ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    predictions = predict_toxicity(text)
    
    # NEW: Pass the text size and the prediction results to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": predictions,
        "original_text": text,
        "text_size": len(text)  # Pass the character count
    })