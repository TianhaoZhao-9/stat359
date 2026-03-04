# make_techwriter_dataset.py
# Creates a Technical Writer persona instruction-tuning dataset in the exact format
# expected by train_tinystories_chat_model.py:
#   {"conversation": [{"text": "<user>"}, {"text": "<assistant>"}]}
#
# It also writes:
#   - train.jsonl
#   - validation.jsonl
#
# Usage (from student/final_project/tinystories_llm):
#   poetry run python .\make_techwriter_dataset.py

import json
import random
from pathlib import Path

random.seed(359)

# --- Prompt pools (match your 20-prompt eval buckets: storytelling, explanation, advice) ---
STORY_PROMPTS = [
    "Tell me a short story about a brave {hero}.",
    "Write a tiny story that includes a {obj1} and a {obj2}.",
    "Tell a bedtime story about a friendly {creature}.",
    "Write a story about a kid who learns {virtue}.",
    "Tell a short story about {theme}.",
    "Write a story about a {small_thing} that wants to be {dream}.",
    "Tell a story set on a {place} during a {weather}.",
    "Write a short story with dialogue between a {char1} and a {char2}.",
]

EXPLAIN_PROMPTS = [
    "Explain what {concept} is in simple terms.",
    "Explain {concept} like I'm new to machine learning.",
    "What's the difference between {concept_a} and {concept_b}?",
    "Explain why overfitting is bad and how to reduce it.",
    "Explain what a tokenizer does.",
    "Explain what perplexity means.",
    "Explain what a checkpoint is in training.",
    "Explain mixed precision (AMP) and why people use it.",
]

ADVICE_PROMPTS = [
    "Give me advice for staying focused while studying.",
    "Give advice for handling stress before an exam.",
    "Give advice for writing clearer emails.",
    "Give advice for presenting a project confidently.",
    "Give advice for improving my sleep schedule.",
    "How can I build a consistent daily routine?",
    "How do I stop procrastinating on a big assignment?",
    "Give advice for giving constructive feedback to a teammate.",
]

# --- Small vocab pools to create variety ---
HEROES = ["cat", "dog", "bunny", "kid", "tiny dragon", "robot", "turtle", "owl", "fox"]
CREATURES = ["shark", "whale", "octopus", "seagull", "dolphin", "crab", "penguin", "bear"]
OBJECTS = ["map", "compass", "lantern", "anchor", "key", "bottle", "coin", "rope", "notebook"]
VIRTUES = ["patience", "kindness", "courage", "honesty", "teamwork", "gratitude"]
THEMES = ["sharing", "teamwork", "helping a sad friend", "being brave", "learning from mistakes"]
SMALL_THINGS = ["little boat", "tiny lantern", "small robot", "mini crab", "tiny compass", "small notebook"]
DREAMS = ["brave", "helpful", "a great explorer", "a good friend", "a captain", "a great helper"]
PLACES = ["island", "dock", "quiet beach", "small village", "forest edge", "cove", "library", "workshop"]
WEATHERS = ["storm", "fog", "rainy night", "windy day", "snowy evening", "sunny morning"]
CHARS = ["captain", "deckhand", "parrot", "mermaid", "shopkeeper", "navigator", "cook", "teacher"]

CONCEPTS = [
    "neural network",
    "gradient descent",
    "attention in transformers",
    "training vs validation loss",
    "learning rate",
    "batch size",
    "sequence length",
    "regularization",
    "early stopping",
    "fine-tuning",
    "instruction tuning",
    "tokenizer",
    "perplexity",
    "checkpoint",
    "mixed precision (AMP)",
]

CONCEPT_PAIRS = [
    ("training loss", "validation loss"),
    ("overfitting", "underfitting"),
    ("tokenizer", "vocabulary"),
    ("epoch", "step"),
    ("fine-tuning", "training from scratch"),
    ("precision", "recall"),
    ("learning rate", "batch size"),
]

# --- Technical Writer style helpers ---
def tw_style_header(title: str) -> str:
    return f"{title}\n"

def tw_bullets(items):
    return "\n".join([f"- {x}" for x in items])

def tw_short_story(hero, obj1=None, obj2=None, creature=None, virtue=None, theme=None,
                   small_thing=None, dream=None, place=None, weather=None, char1=None, char2=None):
    # Keep it readable and structured (still a story, but "technical writer" clarity)
    p = place or random.choice(PLACES)
    w = weather or random.choice(WEATHERS)

    lines = []
    lines.append(f"Story\n")
    lines.append(f"Once upon a time, there was a {hero}.")
    lines.append(f"The {hero} was near a {p} on a {w}.")
    if obj1 and obj2:
        lines.append(f"They found a {obj1} and a {obj2}.")
    if creature:
        lines.append(f"They met a friendly {creature} who offered help.")
    if virtue:
        lines.append(f"The {hero} practiced {virtue} and kept going.")
    if theme:
        lines.append(f"They learned about {theme} by helping someone in need.")
    if small_thing and dream:
        lines.append(f"A {small_thing} wanted to be {dream}, and the {hero} encouraged it.")
    if char1 and char2:
        lines.append(f'"We can do this," said the {char1}. "Yes—together," replied the {char2}.')
    lines.append("In the end, the problem was solved, and everyone felt proud.")
    return " ".join(lines)

def tw_explain(concept: str) -> str:
    # Clear structure: Definition -> Why it matters -> Example/Tip
    if concept in ("training vs validation loss",):
        definition = "Training loss measures error on the training data; validation loss measures error on held-out data."
        why = "Comparing them helps detect overfitting and assess generalization."
        tip = "If training loss keeps decreasing while validation loss increases, consider regularization or early stopping."
        return "\n".join([
            tw_style_header("Explanation"),
            f"Definition: {definition}",
            f"Why it matters: {why}",
            f"Practical tip: {tip}",
        ])

    if concept == "overfitting":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: Overfitting occurs when a model learns training-specific patterns that do not generalize.",
            "Symptoms: Training loss decreases while validation loss stops improving or increases.",
            "Mitigations:",
            tw_bullets([
                "Use more data or data augmentation.",
                "Add regularization (e.g., weight decay, dropout).",
                "Use early stopping based on validation loss."
            ])
        ])

    if concept == "tokenizer":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: A tokenizer converts text into tokens (IDs) that the model can process.",
            "How it works (BPE): It builds a vocabulary of common subword pieces and encodes text as a sequence of those pieces.",
            "Why it matters: Tokenization affects sequence length, vocabulary coverage, and training efficiency."
        ])

    if concept == "perplexity":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: Perplexity is a measure of how well a language model predicts the next token.",
            "Interpretation: Lower perplexity implies the model is less 'surprised' and predicts more confidently.",
            "Note: Perplexity is directly related to cross-entropy loss."
        ])

    if concept == "checkpoint":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: A checkpoint is a saved snapshot of model weights during training.",
            "Why it matters: You can resume training after interruptions and keep the best model based on validation loss.",
            "Common practice: Save checkpoints periodically and select the best checkpoint for deployment."
        ])

    if concept == "mixed precision (AMP)":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: Automatic Mixed Precision (AMP) uses lower precision (e.g., float16) for many operations while keeping stability-critical parts in higher precision.",
            "Benefits: Faster training and lower GPU memory usage.",
            "Caution: Monitor for instability (e.g., NaNs) and adjust settings if needed."
        ])

    if concept == "gradient descent":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: Gradient descent is an optimization method that updates parameters to reduce loss.",
            "Mechanism: It uses the gradient (direction of steepest increase) to move parameters in the opposite direction.",
            "Tuning: The learning rate controls step size—too large can diverge; too small can be slow."
        ])

    if concept == "attention in transformers":
        return "\n".join([
            tw_style_header("Explanation"),
            "Definition: Attention lets the model weigh different tokens when producing the next token.",
            "Why it matters: It captures long-range dependencies by focusing on relevant context.",
            "Result: Better handling of relationships between distant words in a sequence."
        ])

    # Default structured explanation
    return "\n".join([
        tw_style_header("Explanation"),
        f"Definition: {concept} is a core concept used in machine learning workflows.",
        "Why it matters: It influences model training dynamics or generalization.",
        "Practical tip: Track validation loss to ensure improvements carry over to unseen data."
    ])

def tw_compare(a: str, b: str) -> str:
    return "\n".join([
        tw_style_header("Comparison"),
        f"{a.capitalize()}: Measured on the training data; reflects how well the model fits what it has seen.",
        f"{b.capitalize()}: Measured on held-out data; reflects how well the model generalizes.",
        "Rule of thumb: If training improves but validation worsens, the model is likely overfitting."
    ])

def tw_advice(topic: str) -> str:
    # Keep advice structured, actionable, and concise
    if "focused" in topic:
        steps = [
            "Pick a single task and define a clear finish line (e.g., 'write 200 words').",
            "Use a timer (25 minutes work, 5 minutes break).",
            "Remove distractions (phone away, notifications off).",
            "End by writing the next step to reduce restart friction."
        ]
    elif "stress" in topic or "exam" in topic:
        steps = [
            "Use a short breathing reset (inhale 4s, hold 4s, exhale 6s).",
            "Do one targeted review pass; avoid cramming new topics late.",
            "Prioritize sleep—memory consolidation matters.",
            "Start the exam with easier questions to build momentum."
        ]
    elif "emails" in topic:
        steps = [
            "Put the main ask in the first sentence.",
            "Use short paragraphs and bullets for requests.",
            "Include a specific deadline or next action.",
            "Remove filler and keep tone professional."
        ]
    elif "presenting" in topic:
        steps = [
            "Open with a one-sentence summary of your goal.",
            "Keep 3–5 key points and practice transitions.",
            "Pause before answering questions; restate the question briefly.",
            "Close with a clear takeaway."
        ]
    elif "sleep" in topic:
        steps = [
            "Set a fixed wake time and stick to it.",
            "Get light exposure in the morning.",
            "Stop caffeine after early afternoon.",
            "Reduce screens 30 minutes before bed."
        ]
    elif "routine" in topic:
        steps = [
            "Pick one anchor habit (e.g., same wake time).",
            "Add one small behavior immediately after it.",
            "Track daily; review weekly and adjust.",
            "Keep changes small to improve consistency."
        ]
    elif "procrastinating" in topic:
        steps = [
            "Define the smallest possible first step (open the file, write one sentence).",
            "Work for 10 minutes to lower activation energy.",
            "Break tasks into 30–60 minute chunks with specific outputs.",
            "Reward completion of each chunk to reinforce progress."
        ]
    elif "feedback" in topic:
        steps = [
            "Start with what went well (specific example).",
            "State one improvement area and why it matters.",
            "Suggest a concrete next step.",
            "Agree on follow-up to confirm progress."
        ]
    else:
        steps = [
            "Clarify the goal and success criteria.",
            "Break the work into small steps.",
            "Do the first step immediately.",
            "Review progress regularly and adjust."
        ]

    return "\n".join([
        tw_style_header("Advice"),
        tw_bullets(steps)
    ])

def make_example(kind: str, i: int):
    if kind == "story":
        template = random.choice(STORY_PROMPTS)
        user = template.format(
            hero=random.choice(HEROES),
            obj1=random.choice(OBJECTS),
            obj2=random.choice(OBJECTS),
            creature=random.choice(CREATURES),
            virtue=random.choice(VIRTUES),
            theme=random.choice(THEMES),
            small_thing=random.choice(SMALL_THINGS),
            dream=random.choice(DREAMS),
            place=random.choice(PLACES),
            weather=random.choice(WEATHERS),
            char1=random.choice(CHARS),
            char2=random.choice(CHARS),
        )
        assistant = tw_short_story(
            hero=random.choice(HEROES),
            obj1=random.choice(OBJECTS),
            obj2=random.choice(OBJECTS),
            creature=random.choice(CREATURES),
            virtue=random.choice(VIRTUES),
            theme=random.choice(THEMES),
            small_thing=random.choice(SMALL_THINGS),
            dream=random.choice(DREAMS),
            place=random.choice(PLACES),
            weather=random.choice(WEATHERS),
            char1=random.choice(CHARS),
            char2=random.choice(CHARS),
        )

    elif kind == "explain":
        if random.random() < 0.25:
            a, b = random.choice(CONCEPT_PAIRS)
            user = f"What's the difference between {a} and {b}?"
            assistant = tw_compare(a, b)
        else:
            concept = random.choice(CONCEPTS + ["overfitting"])
            template = random.choice(EXPLAIN_PROMPTS)
            user = template.format(
                concept=concept,
                concept_a=random.choice(CONCEPTS),
                concept_b=random.choice(CONCEPTS),
            )
            assistant = tw_explain(concept)

    else:
        user = random.choice(ADVICE_PROMPTS)
        assistant = tw_advice(user)

    # EXACT schema your trainer should accept:
    # - "conversation" field exists
    # - each turn has "text"
    # - strictly alternating user then assistant
    ex = {
        "conversation": [
            {"text": user},
            {"text": assistant},
        ]
    }
    return ex

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    base_dir = Path(__file__).resolve().parent

    n_total = 2000
    n_story = n_total // 3
    n_explain = n_total // 3
    n_advice = n_total - n_story - n_explain

    kinds = (["story"] * n_story) + (["explain"] * n_explain) + (["advice"] * n_advice)
    random.shuffle(kinds)

    all_rows = [make_example(k, i) for i, k in enumerate(kinds, start=1)]

    # 90/10 split
    random.shuffle(all_rows)
    n_val = max(1, int(0.1 * len(all_rows)))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    # Write files
    write_jsonl(base_dir / "techwriter_persona.jsonl", all_rows)
    write_jsonl(base_dir / "train.jsonl", train_rows)
    write_jsonl(base_dir / "validation.jsonl", val_rows)

    print(f"Wrote techwriter_persona.jsonl ({len(all_rows)} examples)")
    print(f"Wrote train.jsonl ({len(train_rows)} examples)")
    print(f"Wrote validation.jsonl ({len(val_rows)} examples)")

if __name__ == "__main__":
    main()