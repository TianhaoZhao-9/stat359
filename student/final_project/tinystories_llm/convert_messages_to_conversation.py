import json
from pathlib import Path

src = Path("pirate_persona.jsonl")
out = Path("pirate_persona_fixed.jsonl")

with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
    for line in f_in:
        obj = json.loads(line)
        msgs = obj.get("messages", [])

        conv = []
        for m in msgs:
            role = m.get("role")
            if role in ("user", "assistant"):
                conv.append({
                    "role": role,
                    "text": m.get("content", "")
                })

        f_out.write(json.dumps({"conversation": conv}, ensure_ascii=False) + "\n")

print("Wrote:", out)