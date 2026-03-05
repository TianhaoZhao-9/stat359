import subprocess
from pathlib import Path

PROMPTS = [
    "Tell me a short story about a brave turtle.",
    "Write a story that includes a lantern and a key.",
    "Tell a bedtime story about a friendly whale.",
    "Write a short story about teamwork.",
    "Tell a story set on a windy day near a dock.",
    "Write a short dialogue between a captain and a navigator.",
    "Tell a story about a robot learning patience.",
    "Explain what overfitting is.",
    "Explain the difference between training loss and validation loss.",
    "What is a neural network?",
    "Explain what a tokenizer does.",
    "What does perplexity measure?",
    "Explain gradient descent.",
    "What is instruction tuning?",
    "How can I stop procrastinating on a big assignment?",
    "Give advice for staying focused while studying.",
    "Give advice for handling stress before an exam.",
    "How can I build a consistent daily routine?",
    "Give advice for presenting a project confidently.",
    "Give advice for writing clearer emails."
]

MODELS = {
    "default": "tinystories_chat_model/best_model.pth",
    "pirate": "pirate_chat_model/final_model.pth",
    "techwriter": "techwriter_chat_model/final_model.pth",
    "moreepochs": "tinystories_chat_model_moreepochs/final_model.pth"
}

BASE_DIR = Path(__file__).resolve().parent

def run_model(model_path, prompt):
    result = subprocess.run(
        ["poetry", "run", "python", "chat_with_tinystories_model.py",
         "--model_path", model_path],
        input=prompt + "\nexit\n",
        text=True,
        capture_output=True
    )
    return result.stdout

for name, model_path in MODELS.items():
    print(f"\nEvaluating {name} model...")
    out_file = BASE_DIR / f"{name}_outputs.txt"
    with out_file.open("w", encoding="utf-8") as f:
        for i, prompt in enumerate(PROMPTS, 1):
            print(f"Prompt {i}: {prompt}")
            output = run_model(model_path, prompt)
            f.write(f"\n--- Prompt {i} ---\n")
            f.write(prompt + "\n\n")
            f.write(output + "\n")