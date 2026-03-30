import re
import unicodedata
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "models/muril_detector"
SAVE_DIR    = "models/v2_muril_detector"
HASOC_PATH  = "data/hasoc2019.tsv"
KAGGLE_PATH = "data/kaggle.csv"

label2id = {"O": 0, "DIRECT": 1, "SEXUAL": 2, "SLUR": 3}
id2label = {0: "O", 1: "DIRECT", 2: "SEXUAL", 3: "SLUR"}

BASE_SENTENCES = [
    # Hindi
    ["मैं", "आज", "काम", "कर", "रहा", "हूँ"],
    ["वह", "बहुत", "बड़ा", "इंसान", "है"],
    ["तुम", "एक", "अच्छे", "इंसान", "हो"],
    ["यह", "आदमी", "यहाँ", "आया", "था"],
    ["उसने", "मुझसे", "बात", "की", "थी"],
    ["मुझे", "आज", "बाहर", "जाना", "है"],
    ["वो", "हमेशा", "ऐसा", "ही", "करता", "है"],
    # English
    ["you", "are", "such", "a", "person"],
    ["he", "is", "really", "something", "else"],
    ["i", "cannot", "believe", "this", "at", "all"],
    ["what", "kind", "of", "person", "are", "you"],
    ["this", "guy", "is", "absolutely", "unbelievable"],
    # Hinglish
    ["yaar", "tu", "toh", "aisa", "hi", "hai"],
    ["bhai", "ye", "kya", "kar", "raha", "hai"],
    ["abe", "sun", "na", "ek", "baar", "meri"],
    ["arey", "tum", "log", "kya", "samajhte", "ho"],
    ["yeh", "banda", "bilkul", "alag", "hai"],
]

# ── Word Lists ────────────────────────────────────────────────────────────────
ABUSE_ROOTS = {
    "DIRECT": {
        "hindi": ["कमीना", "हरामजादा", "साला", "कमबख्त", "रंडी", "कुत्ता"],
        "english": ["idiot","stupid","dumb","moron","bastard","jerk","loser","fool","scum"],
        "hinglish": ["kamina","kameena","harami","haramzada","saala","kutte","randi","bewakoof"],
    },
    "SEXUAL": {
        "hindi": ["भड़वा", "चोद", "चूत", "गांड", "चूतिया"],
        "english": ["fuck","fucking","motherfucker","bitch","slut","whore","asshole","shit"],
        "hinglish": [
            "chutiya","chut","gaand","gaandu","bhenchod","behnchod",
            "madarchod","bhadwa","maderchod","bc","mc","lund","lavde"
        ],
    },
    "SLUR": {
        "hindi": [],
        "english": ["nigger","faggot","retard","spastic"],
        "hinglish": ["chamar","bhangi","jihadi","katua"],
    },
}

SUFFIXES = ["", "े", "ी", "ा", "ों", "ूँगा", "ेगा", "ेगी", "ो", "ु"]

ROMAN_MAP = {
    "कमीना":   ["kamina", "kameena"],
    "भड़वा":    ["bhadwa", "bhadua"],
    "चूत":     ["chut", "ch*t"],
    "गांड":    ["gaand", "g@and"],
    "चूतिया":  ["chutiya", "chut!ya", "ch*tiya"],
    "बहनचोद":  ["bhenchod", "behnchod", "b3henchod"],
    "मादरचोद": ["madarchod", "m@darchod"],
    "हरामजादा":["haramzada", "haramjaada"],
}

# ── Leet Speak Decoder ────────────────────────────────────────────────────────
LEET_MAP = str.maketrans({
    "@": "a", "3": "e", "!": "i", "0": "o",
    "1": "i", "$": "s", "*": "u", "+": "t"
})

def decode_leet(text):
    return text.translate(LEET_MAP)

# ── Text Helpers ──────────────────────────────────────────────────────────────
def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = decode_leet(text)
    return re.sub(r"\s+", " ", text).strip()

def noise_variants(word):
    variants = {word, word.replace(" ", ""), decode_leet(word)}
    if len(word) >= 4:
        variants.add(word[:-1])
        variants.add(word + "a")
        variants.add(word + word[-1])  # repeated last char evasion
    return variants

def expand_roots(roots, category):
    """Expand root words with suffixes, roman variants and noise."""
    expanded = {}
    for lang_roots in roots[category].values():
        for root in lang_roots:
            all_variants = set()
            all_variants |= {root + suf for suf in SUFFIXES}
            all_variants |= set(ROMAN_MAP.get(root, []))
            all_variants |= noise_variants(root)
            for v in all_variants:
                expanded[normalize_text(v)] = category
    return expanded

# Build master word → category lookup
WORD_TO_LABEL = {}
for cat in ["DIRECT", "SEXUAL", "SLUR"]:
    WORD_TO_LABEL.update(expand_roots(ABUSE_ROOTS, cat))

# ── Real Data Loading ─────────────────────────────────────────────────────────
HASOC_TASK2_MAP = {
    "OFFN": "DIRECT",
    "PRFN": "SEXUAL",
    "HATE": "SLUR",
    "NONE": "O",
    "none": "O",
}

def load_hasoc(path):
    df = pd.read_csv(path, sep="\t", usecols=["text", "task_2"])
    df = df.dropna(subset=["text", "task_2"])
    df["category"] = df["task_2"].map(HASOC_TASK2_MAP).fillna("O")
    return df[["text", "category"]]

def load_kaggle(path):
    df = pd.read_csv(path, usecols=["text", "label"])
    df = df.dropna(subset=["text", "label"])
    # Kaggle only has offensive/not offensive — map to DIRECT as default
    df["category"] = df["label"].apply(
        lambda x: "DIRECT" if "offensive" in str(x).lower() and "not" not in str(x).lower() else "O"
    )
    return df[["text", "category"]]

def sentence_to_word_labels(text, sentence_category):
    """
    Convert a sentence-level label to word-level labels.
    Words matching known abuse vocab get sentence_category,
    everything else gets O.
    """
    tokens = normalize_text(text).split()
    labels = []
    for tok in tokens:
        clean = re.sub(r"[^\w\s]", "", tok)  # strip punctuation
        if clean in WORD_TO_LABEL:
            labels.append(WORD_TO_LABEL[clean])
        elif sentence_category != "O" and clean in WORD_TO_LABEL:
            labels.append(sentence_category)
        else:
            labels.append("O")
    return tokens, labels

def load_real_data():
    """
    Load real data from HASOC and Kaggle datasets.
    """
    sentences, labels = [], []
    for load_fn, path in [(load_hasoc, HASOC_PATH), (load_kaggle, KAGGLE_PATH)]:
        try:
            df = load_fn(path)
            for _, row in df.iterrows():
                tokens, lbls = sentence_to_word_labels(row["text"], row["category"])
                if len(tokens) > 0 and len(tokens) == len(lbls):
                    sentences.append(tokens)
                    labels.append(lbls)
            print(f"✅ Loaded {len(df)} rows from {path}")
        except Exception as e:
            print(f"⚠️  Could not load {path}: {e}")
    return sentences, labels

# ── Synthetic Data ────────────────────────────────────────────────────────────
def build_synthetic_data():
    """
    Create synthetic sentences by inserting abuse words into clean base sentences.
    This helps the model learn to detect abuse in various contexts and positions."""
    sentences, labels = [], []

    all_words = list(WORD_TO_LABEL.items())  # (word, category) pairs

    for word, category in all_words:
        for base in BASE_SENTENCES:
            # Insert at multiple positions
            for insert_pos in range(1, min(len(base), 4)):
                tokens = base.copy()
                tokens.insert(insert_pos, word)
                normalized = [normalize_text(t) for t in tokens]
                lbls = ["O"] * len(normalized)
                lbls[insert_pos] = category
                sentences.append(normalized)
                labels.append(lbls)

    # Clean sentences (no abuse)
    for base in BASE_SENTENCES:
        for _ in range(15):
            sentences.append([normalize_text(t) for t in base])
            labels.append(["O"] * len(base))

    # Multi-abuse sentences
    import random
    random.seed(42)
    abuse_words = [(w, c) for w, c in all_words if c != "O"]
    for _ in range(500):
        base = random.choice(BASE_SENTENCES).copy()
        w1, c1 = random.choice(abuse_words)
        w2, c2 = random.choice(abuse_words)
        tokens = base.copy()
        tokens.insert(1, w1)
        tokens.insert(4, w2)
        normalized = [normalize_text(t) for t in tokens]
        lbls = ["O"] * len(normalized)
        if len(lbls) > 1: lbls[1] = c1
        if len(lbls) > 4: lbls[4] = c2
        sentences.append(normalized)
        labels.append(lbls)

    print(f"✅ Built {len(sentences)} synthetic samples")
    return sentences, labels

# ── Tokenization ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(example):
    tokenized = tokenizer(
        example["tokens"], is_split_into_words=True,
        truncation=True, padding="max_length", max_length=128
    )
    label_ids = [
        -100 if wid is None else label2id[example["labels"][wid]]
        for wid in tokenized.word_ids()
    ]
    tokenized["labels"] = label_ids
    return tokenized

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load and merge data
    real_sents, real_labels     = load_real_data()
    synth_sents, synth_labels   = build_synthetic_data()

    all_sentences = real_sents + synth_sents
    all_labels    = real_labels + synth_labels

    print(f"\n📊 Total samples: {len(all_sentences)}")

    # Build dataset
    dataset = (
        Dataset.from_dict({"tokens": all_sentences, "labels": all_labels})
        .map(tokenize_and_align_labels)
        .train_test_split(test_size=0.15, seed=42)
    )

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=4, id2label=id2label, label2id=label2id
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=TrainingArguments(
            output_dir=SAVE_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-6,
            per_device_train_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_steps=100,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    print("\n🚀 Training abuse detector...")
    trainer.train()

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\n✅ Abuse detector saved to {SAVE_DIR}")
