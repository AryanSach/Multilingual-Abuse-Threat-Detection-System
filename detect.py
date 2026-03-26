import re
import unicodedata
import torch
from collections import defaultdict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ───────────────────────────────────────────────────────────────────
DETECTOR_DIR = "models/v2_muril_detector"
THREAT_DIR   = "models/xlmr_threat"

CATEGORY_LABELS = {
    "DIRECT": "direct abuse",
    "SEXUAL": "sexual abuse",
    "SLUR":   "slur",
    "THREAT": "threatening",
}

LEET_MAP = str.maketrans({
    "@": "a", "3": "e", "!": "i", "0": "o",
    "1": "i", "$": "s", "*": "u", "+": "t"
})

# ── Load Models ───────────────────────────────────────────────────────────────
print("Loading abuse detector...")
detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_DIR)
detector_model     = AutoModelForTokenClassification.from_pretrained(DETECTOR_DIR).to(device)
detector_model.eval()

print("Loading threat detector...")
threat_tokenizer = AutoTokenizer.from_pretrained(THREAT_DIR)
threat_model     = AutoModelForSequenceClassification.from_pretrained(THREAT_DIR).to(device)
threat_model.eval()


print("✅ Both models loaded\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.translate(LEET_MAP)
    return re.sub(r"\s+", " ", text).strip()

def reconstruct_word(pieces):
    word = ""
    for piece in pieces:
        word += piece[2:] if piece.startswith("##") else piece
    return word

# ── Abuse Word Detection ──────────────────────────────────────────────────────
def detect_abusive_words(text, threshold=0.5):
    """
    Returns list of (word, category) tuples.
    Categories: DIRECT, SEXUAL, SLUR
    """
    normalized = normalize_text(text)
    inputs = detector_tokenizer(
        normalized, return_tensors="pt",
        truncation=True, max_length=128
    )
    word_ids = inputs.word_ids()
    model_inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = detector_model(**model_inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0]  # [seq_len, num_labels]

    tokens = detector_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    word_probs  = defaultdict(lambda: defaultdict(list))  # wid → label → [probs]
    word_pieces = defaultdict(list)

    id2label = detector_model.config.id2label

    for token, prob, wid in zip(tokens, probs, word_ids):
        if wid is None or token in ["[CLS]", "[SEP]", "<s>", "</s>"]:
            continue
        for label_id, label_name in id2label.items():
            word_probs[wid][label_name].append(prob[label_id].item())
        word_pieces[wid].append(token)

    results = []
    for wid, label_probs in word_probs.items():
        # Find the label with highest average probability
        avg_probs = {
            label: sum(ps) / len(ps)
            for label, ps in label_probs.items()
        }
        best_label = max(avg_probs, key=avg_probs.get)
        best_prob  = avg_probs[best_label]

        if best_label != "O" and best_prob > threshold:
            word = reconstruct_word(word_pieces[wid])
            results.append((word, best_label))

    return results

# ── Threat Detection ──────────────────────────────────────────────────────────
def detect_threat(text, threshold=0.6):
    """
    Returns True if sentence is detected as threatening.
    Higher threshold = less false positives.
    """
    normalized = normalize_text(text)
    inputs = threat_tokenizer(
        normalized, return_tensors="pt",
        truncation=True, max_length=128
    )
    model_inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = threat_model(**model_inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0]

    threat_prob = probs[1].item()  # index 1 = THREAT
    return threat_prob > threshold, round(threat_prob, 3)

# ── Combined Analysis ─────────────────────────────────────────────────────────
def analyze(text, abuse_threshold=0.5, threat_threshold=0.6):
    """
    Full analysis pipeline. Returns structured result dict.
    """
    abusive_words    = detect_abusive_words(text, threshold=abuse_threshold)
    is_threat, t_prob = detect_threat(text, threshold=threat_threshold)

    return {
        "text":          text,
        "abusive_words": abusive_words,   # [(word, category), ...]
        "is_threat":     is_threat,
        "threat_score":  t_prob,
    }

def print_results(result):
    """Pretty print analysis results."""
    print(f"\n📝 Text: {result['text']}")
    print("─" * 60)

    if not result["abusive_words"] and not result["is_threat"]:
        print("✅ No abusive content detected")
        return

    if result["abusive_words"]:
        print("🚨 Abusive words detected:")
        for word, category in result["abusive_words"]:
            label = CATEGORY_LABELS.get(category, category.lower())
            print(f"   {word} — {label}")

    if result["is_threat"]:
        print(f"⚠️  Threatening content detected (confidence: {result['threat_score']})")

    print()

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        "आज सुबह सब ठीक था लेकिन फिर वह आदमी आया।",
        "you are such an idiot and a complete moron",
        "bhai tu pagal hai bilkul, tujhe jaan se maar dunga",
        "yaar aaj ka din bahut acha tha",
        "kutte tujhe nahi chhodunga",
        "kya haal hai dost, sab theek toh hai?",
        "तुझे देख लूँगा, तेरी खैर नहीं सााला`",
        "तू मुझसे बच नहीं पाएगा, तेरे घर आऊंगा",
        
    ]

    for text in test_cases:
        result = analyze(text)
        print_results(result)