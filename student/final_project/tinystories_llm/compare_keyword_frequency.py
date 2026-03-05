import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILES = {
    "default": BASE_DIR / "default_outputs_more.txt",
    "default_more_epochs": BASE_DIR / "moreepochs_outputs_more.txt",
    "pirate": BASE_DIR / "pirate_outputs_more.txt",
    "techwriter": BASE_DIR / "techwriter_outputs_more.txt",
}

PIRATE_KEYWORDS = [
    "arr", "matey", "aye", "ahoy", "yo-ho",
    "sail", "treasure", "crew", "ship"
]

TECH_KEYWORDS = [
    "explain",
    "definition",
    "concept",
    "process",
    "step",
    "example",
    "system",
    "function",
    "method",
    "information",
    "data",
    "analysis",
    "result",
    "structure",
    "implementation"
]

def clean_text(text):
    text = text.lower()
    return text

def count_keywords(text, keywords):
    count = 0
    for kw in keywords:
        count += text.count(kw)
    return count

def average_sentence_length(text):
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        return 0

    total_words = 0
    for s in sentences:
        words = re.findall(r"\b\w+\b", s)
        total_words += len(words)

    return total_words / len(sentences)

for name, path in FILES.items():
    if not path.exists():
        print(f"{name}: file not found\n")
        continue

    raw_text = path.read_text(encoding="utf-8")
    text = clean_text(raw_text)

    words = re.findall(r"\b\w+\b", text)
    total_words = len(words)

    pirate_count = count_keywords(text, PIRATE_KEYWORDS)
    tech_count = count_keywords(text, TECH_KEYWORDS)

    pirate_ratio = pirate_count / total_words if total_words > 0 else 0
    tech_ratio = tech_count / total_words if total_words > 0 else 0

    avg_sent_len = average_sentence_length(text)

    print(f"\n===== {name.upper()} =====")
    print(f"Total words: {total_words}")
    print(f"Average sentence length: {avg_sent_len:.2f} words")
    print(f"Pirate keyword count: {pirate_count}")
    print(f"Pirate keyword ratio: {pirate_ratio:.4f}")
    print(f"Tech keyword count: {tech_count}")
    print(f"Tech keyword ratio: {tech_ratio:.4f}")