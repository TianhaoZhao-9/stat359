
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