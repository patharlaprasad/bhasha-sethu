# backend/app.py

import os, re, time, math
import numpy as np
import faiss
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
from sentence_transformers import SentenceTransformer

# ---------------------------
# Flask setup
# ---------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------
# Load Hugging Face token
# ---------------------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing. Set it in backend/.env")

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "te": "Telugu"}

# ---------------------------
# Language code mapping for M2M100
# ---------------------------
LANG_CODE_MAP = {
    "en": "en",
    "hi": "hi",
    "te": "te"  # ✅ FIX: M2M100_418M model uses 'te' for Telugu, not 'te_IN'
}

# ---------------------------
# Translation models
# ---------------------------
def load_pair(model_name):
    tok = MarianTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    mdl = MarianMTModel.from_pretrained(model_name, token=HF_TOKEN)
    return tok, mdl

TOK_EN_HI, MOD_EN_HI = load_pair("Helsinki-NLP/opus-mt-en-hi")
TOK_HI_EN, MOD_HI_EN = load_pair("Helsinki-NLP/opus-mt-hi-en")

M2M_MODEL = "facebook/m2m100_418M"
M2M_TEMP = M2M100Tokenizer.from_pretrained(M2M_MODEL, token=HF_TOKEN)
M2M_MOD = M2M100ForConditionalGeneration.from_pretrained(M2M_MODEL, token=HF_TOKEN)

def translate(text, src, tgt):
    """Translate between English, Hindi, Telugu using Marian + M2M100"""
    
    # Handle English ↔ Hindi via Marian
    if src in ["en", "hi"] and tgt in ["en", "hi"]:
        if src == "en" and tgt == "hi":
            tok, mdl = TOK_EN_HI, MOD_EN_HI
        elif src == "hi" and tgt == "en":
            tok, mdl = TOK_HI_EN, MOD_HI_EN
        else:
            return text  # unsupported pair
        batch = tok([text], return_tensors="pt", padding=True)
        gen = mdl.generate(**batch)
        return tok.batch_decode(gen, skip_special_tokens=True)[0]

    # Use M2M100 if Telugu involved
    src_lang = LANG_CODE_MAP.get(src, src)
    tgt_lang = LANG_CODE_MAP.get(tgt, tgt)

    # Set source language for M2M tokenizer
    M2M_TEMP.src_lang = src_lang

    # Encode text
    encoded = M2M_TEMP(text, return_tensors="pt")

    # Generate translation
    gen = M2M_MOD.generate(
        **encoded,
        forced_bos_token_id=M2M_TEMP.get_lang_id(tgt_lang),
        max_new_tokens=256
    )

    return M2M_TEMP.batch_decode(gen, skip_special_tokens=True)[0]
# ---------------------------
# Hinglish / Tinglish lexicons
# ---------------------------
HINGLISH_LEX = {
    r"\bnamaste\b": "नमस्ते",
    r"\bkya\b": "क्या",
    r"\bkaise\b": "कैसे",
    r"\bha(i|ee)\b": "भाई",
    r"\bmera\b": "मेरा",
    r"\bnaam\b": "नाम",
    r"\bghar\b": "घर",
    r"\bkhana\b": "खाना",
    r"\bthik\b": "ठीक",
    r"\bdost\b": "दोस्त",
    r"\bpadhai\b": "पढ़ाई",
    r"\bschool\b": "स्कूल",
    r"\boffice\b": "ऑफिस",
    r"\bkyu\b": "क्यों",
    r"\bkahan\b": "कहाँ",
    r"\bkab\b": "कब",
    r"\bha\b": "है",
    r"\bhoon\b": "हूँ",
    r"\braha\b": "रहा",
    r"\brahe\b": "रहे",
    r"\bha(i)?\s+na\b": "है ना",
}

TINGLISH_LEX = {
    r"\bnuvvu\b": "నువ్వు",
    r"\bmeeru\b": "మీరు",
    r"\bthinnava\b": "తిన్నావా",
    r"\btinava\b": "తిన్నావా",
    r"\banna\b": "అన్నా",
    r"\bcheppa\b": "చెప్ప",
    r"\bchelli\b": "చెల్లి",
    r"\bbagunna(va|ra)\b": "బాగున్నావా",
    r"\bemiti\b": "ఏమిటీ",
    r"\benduku\b": "ఎందుకు",
    r"\bevaru\b": "ఎవరు",
    r"\bpani\b": "పని",
    r"\bbaga\b": "బాగా",
    r"\bchala\b": "చాలా",
}

def normalize_hinglish(text: str) -> str:
    out = text
    for pat, dev in HINGLISH_LEX.items():
        out = re.sub(pat, dev, out, flags=re.IGNORECASE)
    return out

def normalize_tinglish(text: str) -> str:
    out = text
    for pat, tel in TINGLISH_LEX.items():
        out = re.sub(pat, tel, out, flags=re.IGNORECASE)
    return out

# ---------------------------
# Language detection
# ---------------------------
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")   # Hindi
RE_TELUGU     = re.compile(r"[\u0C00-\u0C7F]")   # Telugu

HINGLISH_CLUES = set("aap tum kya kaise kahan kab kyu mera ghar khana bhai dost school office thik".split())
TINGLISH_CLUES = set("nuvvu meeru thinnava tinava anna chelli cheppa bagunnava emiti enduku evaru pani baga chala".split())

def detect_lang(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "en"
    if RE_DEVANAGARI.search(t):
        return "hi"
    if RE_TELUGU.search(t):
        return "te"
    toks = re.findall(r"[a-zA-Z]+", t.lower())
    if toks:
        h = sum(1 for x in toks if x in HINGLISH_CLUES)
        tg = sum(1 for x in toks if x in TINGLISH_CLUES)
        if h >= max(2, math.ceil(len(toks) * 0.25)):
            return "hinglish"
        if tg >= max(2, math.ceil(len(toks) * 0.25)):
            return "tinglish"
    return "en"

# ---------------------------
# Knowledge Base + RAG
# ---------------------------
KB = [
    {"domain": "health", "lang": "en", "text": "Government primary health centers offer free screenings for diabetes and hypertension to adults over 30."},
    {"domain": "health", "lang": "hi", "text": "सरकारी प्राथमिक स्वास्थ्य केंद्र 30 वर्ष से अधिक आयु के वयस्कों के लिए मधुमेह और उच्च रक्तचाप की मुफ्त जांच प्रदान करते हैं।"},
    {"domain": "health", "lang": "te", "text": "ప్రభుత్వ ప్రాథమిక ఆరోగ్య కేంద్రాలు 30 ఏళ్లు పైబడిన పెద్దలకు ఉచిత మధుమేహం మరియు రక్తపోటు పరీక్షలు అందిస్తాయి."},
    # ... (keep your KB items from before)
]

EMBEDDER = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

KB_TEXTS = [x["text"] for x in KB]
KB_EMB = _normalize(EMBEDDER.encode(KB_TEXTS, convert_to_numpy=True, show_progress_bar=False).astype("float32"))
DIM = KB_EMB.shape[1]
INDEX = faiss.IndexFlatIP(DIM)
INDEX.add(KB_EMB)

def rag_search(query_en: str, top_k=3, threshold=0.45):
    q = EMBEDDER.encode([query_en], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q = _normalize(q)
    D, I = INDEX.search(q, top_k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0: continue
        if float(score) >= threshold:
            item = KB[idx]
            out.append({
                "domain": item["domain"],
                "lang": item["lang"],
                "text": item["text"],
                "score": float(score)
            })
    return out

# ---------------------------
# Helpers
# ---------------------------
def clamp(s: str, n=300):
    s = s.strip()
    return s if len(s) <= n else s[:n] + "…"

def synthesize_answer(retrieved):
    if not retrieved:
        return "I could not find relevant information in the knowledge base."
    bullets = [f"- ({r['domain']}/{r['lang']}) {clamp(r['text'], 220)}" for r in retrieved]
    return "Here is what I found:\n" + "\n".join(bullets)

# ---------------------------
# API
# ---------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return {"status": "ok", "message": "Flask is running!"}

@app.route("/api/process", methods=["POST"])
def api_process():
    t0 = time.time()
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    target_lang = (data.get("target_lang") or "").strip().lower()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    detected = detect_lang(text)
    normalized = text
    if detected == "hinglish":
        normalized = normalize_hinglish(text); detected = "hi"
    elif detected == "tinglish":
        normalized = normalize_tinglish(text); detected = "te"

    query_en = translate(normalized, detected, "en") if detected in ["hi", "te"] else normalized
    retrieved = rag_search(query_en)
    answer_en = synthesize_answer(retrieved)

    out_lang = target_lang if target_lang in SUPPORTED_LANGUAGES else detected
    out_text = translate(answer_en, "en", out_lang) if out_lang != "en" else answer_en

    latency_ms = int((time.time() - t0) * 1000)
    metrics = {
    "latency_ms": latency_ms,
    "detected_lang": detected,
    "target_lang": out_lang,
    "num_retrieved": len(retrieved),
    "bleu": 0.0,
    "comet": 0.0,
    "named_entity_preservation": 0.0,
    "toxicity_leakage": 0.0
    }

    return jsonify({
    "original_text": text,
    "detected_language_name": SUPPORTED_LANGUAGES.get(detected, "English"),
    "normalized_text": normalized if normalized != text else "",
    "retrieved_items": retrieved,
    "translated_text": out_text,
    "metrics": metrics
    })

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
