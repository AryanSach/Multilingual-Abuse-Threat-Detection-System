import re
import unicodedata
import random
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "xlm-roberta-base"
SAVE_DIR    = "models/xlmr_threat"
HASOC_PATH  = "data/hasoc2019.tsv"

label2id = {"NOT_THREAT": 0, "THREAT": 1}
id2label = {0: "NOT_THREAT", 1: "THREAT"}

random.seed(42)

# ── Threat Templates ──────────────────────────────────────────────────────────
# Hindi threats
HINDI_THREATS = [
    "तुझे जान से मार दूँगा",
    "तुझे देख लूँगा",
    "तेरी खैर नहीं",
    "मैं तुझे नहीं छोडूँगा",
    "तू मुझसे बच नहीं पाएगा",
    "तेरा हिसाब करूँगा",
    "तुझे ऐसा सबक सिखाऊंगा जो जिंदगी भर याद रहेगा",
    "अंजाम भुगतना पड़ेगा",
    "तू मेरे साथ ऐसा करेगा तो मैं चुप नहीं बैठूँगा",
    "बहुत पछताएगा",
    "तेरे घर आऊंगा",
    "ध्यान रखना वरना अच्छा नहीं होगा",
    "मैं तुझे और तेरे परिवार को नहीं छोडूँगा",
    "तेरी हड्डियाँ तोड़ दूँगा",
    "मार दूँगा साले",
    "गला दबा दूँगा",
    "तू जहाँ भी जाएगा मैं ढूंढ लूँगा",
]

# Hinglish threats
HINGLISH_THREATS = [
    "tujhe jaan se maar dunga",
    "dekh lena tujhe",
    "teri khair nahi",
    "nahi chhodunga tujhe",
    "tu bach nahi payega",
    "tera hisaab karunga",
    "itna marunga ki yaad rahega",
    "ghar pe aaunga tere",
    "poori family ko nahi chhodunga",
    "tujhe utha ke phenk dunga",
    "ankhe nikal lunga teri",
    "tu samajhta kya hai apne aap ko maar dunga",
    "abhi address de de tera",
    "teri maa ko bhi nahi chhodunga",
    "aa ja saamne bata dunga kya hota hai",
    "marke rakhega tera koi nahi",
    "tujhe uda dunga",
    "seedha hospital pahunchaunga tujhe",
]

# English threats
ENGLISH_THREATS = [
    "i will kill you",
    "watch your back",
    "you won't get away with this",
    "i know where you live",
    "i will find you",
    "you are going to regret this",
    "i will make you pay",
    "your days are numbered",
    "i will destroy you",
    "come outside and say that",
    "i will hurt you and your family",
    "you better be careful",
    "i will end you",
    "say that to my face i dare you",
    "you have no idea what's coming",
    "stay away or else",
    "i will make sure you suffer",
]

# Non-threat sentences (negative examples)
NON_THREATS = [
    # Hindi
    "आज मौसम बहुत अच्छा है",
    "मुझे यह पसंद नहीं आया",
    "तुम गलत हो",
    "यह बहुत बुरा है",
    "मैं तुमसे नाराज हूँ",
    "तुम बेकार हो",
    "यह काम ठीक से करो",
    "मुझे तुमसे बात नहीं करनी",
    "तुम कुछ नहीं जानते",
    "बकवास बंद करो",
    # Hinglish
    "yaar tu pagal hai",
    "bhai ye sahi nahi hai",
    "mujhe pasand nahi yeh",
    "tu galat hai bilkul",
    "aisa mat kar dobara",
    "chup kar yaar",
    "bekar hai tu",
    "kuch nahi pata tujhe",
    "baat karna band kar",
    "tu samajhta kya hai apne aap ko",
    # English
    "i don't like you",
    "you are wrong about this",
    "this is really bad",
    "i am angry at you",
    "you don't know anything",
    "stop talking nonsense",
    "you are useless",
    "i disagree with you",
    "this behavior is unacceptable",
    "you should be ashamed of yourself",
]

def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", " ", text).strip()

def augment_threat(sentence):
    """Create slight variations of threat sentences for data augmentation."""
    variants = [sentence]
    prefixes = ["sun ", "abe ", "oye ", "dekh ", ""]
    suffixes = [" samjha", " teri maa ki", " sala", " kamine", ""]
    for pre in prefixes:
        for suf in suffixes:
            v = (pre + sentence + suf).strip()
            if v != sentence:
                variants.append(v)
    return variants[:5]  # cap at 5 variants per sentence

def build_threat_dataset():
    texts, labels = [], []

    # Add all threat sentences with augmentation
    all_threats = HINDI_THREATS + HINGLISH_THREATS + ENGLISH_THREATS
    for threat in all_threats:
        for variant in augment_threat(normalize_text(threat)):
            texts.append(variant)
            labels.append(1)

    # Add non-threats
    for non_threat in NON_THREATS:
        texts.append(normalize_text(non_threat))
        labels.append(0)

    return texts, labels

def load_hasoc_threats(path):
    """
    Extract potential threat sentences from HASOC.
    HASOC doesn't have explicit threat labels so we use HATE category
    sentences as additional negative/positive signal carefully.
    """
    texts, labels = [], []
    try:
        df = pd.read_csv(path, sep="\t", usecols=["text", "task_1"])
        df = df.dropna()
        # NOT sentences are clean non-threats
        not_rows = df[df["task_1"] == "NOT"].sample(n=min(200, len(df)), random_state=42)
        for text in not_rows["text"]:
            texts.append(normalize_text(text))
            labels.append(0)
        print(f"✅ Loaded {len(not_rows)} non-threat sentences from HASOC")
    except Exception as e:
        print(f"⚠️  Could not load HASOC: {e}")
    return texts, labels

# ── Tokenization ──────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Build dataset
    synth_texts, synth_labels   = build_threat_dataset()
    hasoc_texts, hasoc_labels   = load_hasoc_threats(HASOC_PATH)

    all_texts  = synth_texts + hasoc_texts
    all_labels = synth_labels + hasoc_labels

    print(f"\n📊 Total samples: {len(all_texts)}")
    print(f"   Threats:     {sum(all_labels)}")
    print(f"   Non-threats: {len(all_labels) - sum(all_labels)}")

    dataset = (
        Dataset.from_dict({"text": all_texts, "labels": all_labels})
        .map(tokenize)
        .train_test_split(test_size=0.15, seed=42)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
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
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            num_train_epochs=20,
            weight_decay=0.01,
            logging_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
    )

    print("\n🚀 Training threat detector...")
    trainer.train()

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\n✅ Threat detector saved to {SAVE_DIR}")
