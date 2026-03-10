from pathlib import Path
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

FILES = {
    "default": "default_outputs.txt",
    "pirate": "pirate_outputs.txt",
    "techwriter": "techwriter_outputs.txt",
    "moreepochs": "moreepochs_outputs.txt",
}

REFERENCE = {
    "default": "A neutral assistant answers questions clearly without a strong persona.",
    "pirate": "Arrr matey! A pirate speaks with sea language about ships, treasure, and adventure.",
    "techwriter": "A technical writer explains concepts clearly using structured definitions and precise language.",
    "moreepochs": "A neutral assistant answers questions clearly without a strong persona."
}

def extract_responses(file):
    lines = Path(file).read_text(encoding="utf-8", errors="ignore").splitlines()
    responses = []

    for line in lines:
        if "Assistant:" in line:
            resp = line.split("Assistant:")[1].strip()
            responses.append(resp)

    return responses

model = SentenceTransformer("all-MiniLM-L6-v2")

ref_emb = {k: model.encode(v, convert_to_tensor=True) for k,v in REFERENCE.items()}

rows = []

for model_name, file in FILES.items():
    responses = extract_responses(file)

    sims = []

    for r in responses:
        emb = model.encode(r, convert_to_tensor=True)
        sim = util.cos_sim(emb, ref_emb[model_name]).item()
        sims.append(sim)

    rows.append({
        "Model": model_name,
        "Avg Similarity": sum(sims)/len(sims)
    })

df = pd.DataFrame(rows)
print(df)