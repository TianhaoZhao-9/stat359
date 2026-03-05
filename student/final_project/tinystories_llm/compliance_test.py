import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

def split_sentences(text: str) -> List[str]:
    # simple sentence splitter (good enough for TinyStories outputs)
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]

def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))

def detect_task(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in ["tell me a short story", "write a short story", "write a tiny story", "tell a story", "bedtime story", "once upon"]):
        return "story"
    if any(k in p for k in ["explain", "what is", "what does", "difference between", "define", "means"]):
        return "explain"
    if any(k in p for k in ["give me advice", "how can i", "how do i", "what should i do", "tips for"]):
        return "advice"
    if any(k in p for k in ["answer with one word", "one word"]):
        return "one_word"
    return "other"

def one_word_compliance(prompt: str, response: str) -> Tuple[bool, str]:
    # strict: response should be exactly one word (letters only)
    r = response.strip()
    words = re.findall(r"[A-Za-z]+", r)
    ok = (len(words) == 1) and (r.lower() == words[0].lower())
    return ok, f"words={len(words)}"

def story_compliance(response: str) -> Tuple[bool, str]:
    # basic story compliance: >=3 sentences OR >=35 words
    sents = split_sentences(response)
    words = count_words(response)
    ok = (len(sents) >= 3) or (words >= 35)
    return ok, f"sentences={len(sents)}, words={words}"

def explain_compliance(response: str) -> Tuple[bool, str]:
    # basic explanation compliance:
    # - contains explanation cues and has enough substance (>=2 sentences OR >=20 words)
    cues = ["is", "means", "refers", "because", "so that", "in simple", "for example", "it helps", "it measures"]
    rlow = response.lower()
    has_cue = any(c in rlow for c in cues)
    sents = split_sentences(response)
    words = count_words(response)
    ok = has_cue and ((len(sents) >= 2) or (words >= 20))
    return ok, f"has_cue={has_cue}, sentences={len(sents)}, words={words}"

def advice_compliance(response: str) -> Tuple[bool, str]:
    # advice compliance:
    # - contains at least one suggestion marker + enough length (>=2 sentences OR >=18 words)
    markers = ["should", "try", "can", "recommend", "tip", "start", "avoid", "focus", "set", "make", "plan"]
    rlow = response.lower()
    has_marker = any(m in rlow for m in markers)
    sents = split_sentences(response)
    words = count_words(response)
    ok = has_marker and ((len(sents) >= 2) or (words >= 18))
    return ok, f"has_marker={has_marker}, sentences={len(sents)}, words={words}"

def task_compliance(prompt: str, response: str) -> Tuple[bool, str, str]:
    task = detect_task(prompt)

    # one-word overrides everything if prompt asks it
    if "answer with one word" in prompt.lower() or "one word" in prompt.lower():
        ok, detail = one_word_compliance(prompt, response)
        return ok, task, detail

    if task == "story":
        ok, detail = story_compliance(response)
        return ok, task, detail
    if task == "explain":
        ok, detail = explain_compliance(response)
        return ok, task, detail
    if task == "advice":
        ok, detail = advice_compliance(response)
        return ok, task, detail

    # fallback: at least non-empty and not just echoing the prompt
    ok = bool(response.strip()) and (response.strip().lower() != prompt.strip().lower())
    return ok, task, "fallback_rule"





import re
import pandas as pd
from pathlib import Path

# -------------------------
# 1. Parse assistant outputs
# -------------------------

def extract_responses(file_path):
    lines = Path(file_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    
    responses = []
    current_prompt = None
    
    for line in lines:
        if line.startswith("--- Prompt"):
            current_prompt = int(line.split()[2])
        
        if "Assistant:" in line:
            idx = line.find("Assistant:")
            response = line[idx + len("Assistant:"):].strip()
            responses.append((current_prompt, response))
    
    return responses


# -------------------------
# 2. Utility functions
# -------------------------

def count_sentences(text):
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

def count_words(text):
    return len(re.findall(r"[A-Za-z]+", text))

def contains_advice_word(text):
    advice_words = ["should", "try", "can", "recommend", "focus", "avoid", "plan", "start"]
    text = text.lower()
    return any(word in text for word in advice_words)


# -------------------------
# 3. Compliance logic
# -------------------------

def task_compliance(prompt_id, response):
    sentences = count_sentences(response)
    words = count_words(response)
    
    # Story prompts: 1–7
    if 1 <= prompt_id <= 7:
        return (sentences >= 3) or (words >= 35)
    
    # Explain prompts: 8–14
    elif 8 <= prompt_id <= 14:
        return (sentences >= 2) and (words >= 20)
    
    # Advice prompts: 15–20
    elif 15 <= prompt_id <= 20:
        return (sentences >= 2) and contains_advice_word(response)
    
    return False


# -------------------------
# 4. Evaluate all models
# -------------------------

FILES = {
    "default": "default_outputs.txt",
    "more_epoch": "more_epoch_outputs.txt",
    "pirate": "pirate_outputs.txt",
    "techwriter": "techwriter_outputs.txt",
}

results = []

for model_name, file_path in FILES.items():
    responses = extract_responses(file_path)
    
    total = len(responses)
    compliant = 0
    
    for pid, resp in responses:
        if task_compliance(pid, resp):
            compliant += 1
    
    compliance_rate = compliant / total
    
    results.append({
        "Model": model_name,
        "Compliance Rate": round(compliance_rate, 3),
        "Compliant / Total": f"{compliant}/{total}"
    })

df = pd.DataFrame(results)
print(df)