# app/app.py
from flask import Flask, render_template, request
import torch
import torch.nn.functional as F

from app.model_architecture import BERT, SentenceEncoder, SBERTClassifier

# -------------------------
# Config / constants
# -------------------------
pad_id = 0
cls_id = 1
sep_id = 2

vocab_size = 44623      # ปรับให้ตรงกับ notebook ของคุณ
d_model = 256
d_k = 64
d_v = 64
d_ff = d_model * 4
n_layers = 2
n_heads = 4
max_len = 128
n_segments = 2

# -------------------------
# Build model and load weights
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert = BERT(
    vocab_size=vocab_size,
    d_model=d_model,
    d_k=d_k,
    d_v=d_v,
    d_ff=d_ff,
    n_layers=n_layers,
    n_heads=n_heads,
    max_len=max_len,
    n_segments=n_segments,
    pad_id=pad_id,
)
bert_ckpt = torch.load("model/bert_pretrained_from_scratch.pt", map_location="cpu")
bert.load_state_dict(bert_ckpt)

encoder = SentenceEncoder(bert, cls_id=cls_id, sep_id=sep_id, pad_id=pad_id)

sbert = SBERTClassifier(encoder, hidden_dim=d_model)
sbert_ckpt = torch.load("model/sbert_sentence_encoder.pt", map_location="cpu")
sbert.load_state_dict(sbert_ckpt)

sbert = sbert.to(device)
sbert.eval()

# -------------------------
# Tokenizer (เหมือนใน notebook)
# -------------------------

# NOTE: ในเว็บจริงคุณต้องโหลด word2id/id2word มาด้วย
# ตอนนี้ผมใส่ placeholder ไว้ให้ก่อน
# คุณสามารถเซฟ word2id จาก notebook เป็น .pt หรือ .json แล้วมาโหลดที่นี่

import re
import json

with open("dataset/word2id.json", "r", encoding="utf-8") as f:
    word2id = json.load(f)

unk_id = word2id.get("[UNK]", pad_id)

def encode_sentence(text, max_len=128):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ,.!?'-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for w in text.split():
        if w in word2id:
            tokens.append(word2id[w])
        else:
            tokens.append(unk_id)
    if len(tokens) > max_len - 2:
        tokens = tokens[:max_len - 2]
    return tokens

def make_batch_from_text(premise, hypothesis):
    ids_a = encode_sentence(premise, max_len)
    ids_b = encode_sentence(hypothesis, max_len)

    # แปลงเป็น tensor shape [1, L]
    input_ids_a = torch.tensor([ids_a], dtype=torch.long)
    input_ids_b = torch.tensor([ids_b], dtype=torch.long)

    return input_ids_a.to(device), input_ids_b.to(device)

# -------------------------
# Flask app
# -------------------------

app = Flask(__name__)

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prob_str = None
    premise = ""
    hypothesis = ""

    if request.method == "POST":
        premise = request.form.get("premise", "")
        hypothesis = request.form.get("hypothesis", "")

        if premise.strip() and hypothesis.strip():
            with torch.no_grad():
                inp_a, inp_b = make_batch_from_text(premise, hypothesis)
                logits = sbert(inp_a, inp_b)
                probs = F.softmax(logits, dim=-1)[0]  # [3]

                pred_id = int(torch.argmax(probs).item())
                prediction = LABEL_MAP.get(pred_id, str(pred_id))

                prob_str = {
                    "entailment": float(probs[0].item()),
                    "neutral": float(probs[1].item()),
                    "contradiction": float(probs[2].item()),
                }

    return render_template(
        "index.html",
        prediction=prediction,
        prob_str=prob_str,
        premise=premise,
        hypothesis=hypothesis,
    )

if __name__ == "__main__":
    app.run(debug=True)
