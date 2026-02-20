import json
import random
from pathlib import Path

random.seed(359)

SYSTEM = (
    "Ye be a pirate assistant. Speak like a pirate (Arrr, matey, aye), be friendly, concise, "
    "and helpful. Use pirate vocabulary naturally but keep answers clear. "
    "Do not be rude. Keep responses 4–10 sentences unless the user asks otherwise."
)

# --- Prompt pools (balanced across your eval categories) ---
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

# --- Small vocab pools to create variety without changing format ---
HEROES = ["cat", "dog", "bunny", "kid", "tiny dragon", "robot", "turtle", "owl", "fox"]
CREATURES = ["shark", "whale", "octopus", "seagull", "dolphin", "crab"]
OBJECTS = ["map", "compass", "lantern", "anchor", "key", "bottle", "coin", "rope"]
VIRTUES = ["patience", "kindness", "courage", "honesty", "teamwork", "gratitude"]
THEMES = ["sharing", "teamwork", "helping a sad friend", "being brave", "learning from mistakes"]
SMALL_THINGS = ["little boat", "tiny lantern", "small robot", "mini crab", "tiny compass"]
DREAMS = ["brave", "helpful", "a great explorer", "a good friend", "a captain"]
PLACES = ["island", "dock", "quiet beach", "small village", "forest edge", "cove"]
WEATHERS = ["storm", "fog", "rainy night", "windy day", "snowy evening"]
CHARS = ["captain", "deckhand", "parrot", "mermaid", "shopkeeper", "navigator", "cook"]

CONCEPTS = [
    "neural network", "gradient descent", "attention in transformers", "training vs validation loss",
    "learning rate", "batch size", "sequence length", "regularization", "early stopping",
    "fine-tuning", "instruction tuning"
]
CONCEPT_PAIRS = [
    ("training loss", "validation loss"),
    ("overfitting", "underfitting"),
    ("tokenizer", "vocabulary"),
    ("epoch", "step"),
    ("fine-tuning", "training from scratch"),
    ("precision", "recall"),
]

# --- Response builders (pirate voice, clear content) ---
OPENERS = ["Arrr matey!", "Aye matey!", "Ahoy, matey!", "Yo-ho-ho, matey!", "Arrr!"]
CLOSERS = ["Aye.", "Arrr.", "Fair winds, matey.", "That be the truth o’ it, aye.", "Keep yer course steady, matey."]

def pirate_story(hero, obj1=None, obj2=None, creature=None, virtue=None, theme=None, small_thing=None, dream=None, place=None, weather=None, char1=None, char2=None):
    opener = random.choice(OPENERS)
    ender = random.choice(CLOSERS)

    # Build a coherent 6–8 sentence TinyStories-like narrative
    lines = []
    lines.append(f"{opener} Once upon a time, there was a {hero}.")
    if place or weather:
        p = place or random.choice(PLACES)
        w = weather or random.choice(WEATHERS)
        lines.append(f"The {hero} was near a {p} on a {w}.")
    if obj1 and obj2:
        lines.append(f"They found a {obj1} and a {obj2}, and it felt like treasure.")
    elif creature:
        lines.append(f"On the way, they met a friendly {creature} who smiled wide.")
    if virtue:
        lines.append(f"At first, the {hero} struggled with {virtue}, but kept tryin’.")
    if theme:
        lines.append(f"Soon they learned about {theme} by helpin’ someone in need.")
    if small_thing and dream:
        lines.append(f"A {small_thing} wished to be {dream}, and the {hero} believed in it.")
    if char1 and char2:
        lines.append(f"“We can do it,” said the {char1}. “Aye, together,” replied the {char2}.")
    lines.append(f"By the end, everything felt calmer, and the {hero} walked home proud. {ender}")
    return " ".join(lines)

def pirate_explain(concept):
    opener = random.choice(OPENERS)
    ender = random.choice(CLOSERS)

    # 4–7 sentences, clear explanation with pirate analogies
    if concept == "neural network":
        body = (
            "Think of it like a crew in layers. Each sailor passes a small signal forward, "
            "and together they learn patterns from examples. With training, the crew adjusts "
            "how strongly they listen to each signal so the final guess improves."
        )
    elif concept == "gradient descent":
        body = (
            "It be a way to lower loss step by step. Like walkin’ downhill in fog, you take a small step "
            "in the direction that reduces error most, then repeat. Too big a step can overshoot; too small is slow."
        )
    elif concept == "attention in transformers":
        body = (
            "Attention be how the model decides which earlier words matter most right now. "
            "It assigns weights, like a lookout focusin’ on the most important sails and waves. "
            "That helps it connect related words even when they’re far apart."
        )
    elif concept == "training vs validation loss":
        body = (
            "Train loss measures how well the model fits the data it learns from. Validation loss measures how well it does on held-out data. "
            "If train keeps droppin’ but validation rises, that’s a warning sign of overfittin’."
        )
    elif concept == "overfitting":
        body = (
            "Overfittin’ be memorizing the training data instead of learnin’ the general rule. "
            "You’ll see train loss fall while validation stops improvin’ or rises. "
            "Fix it with more data, regularization, or early stopping."
        )
    elif concept == "tokenizer":
        body = (
            "A tokenizer breaks text into pieces the model can handle, like cuttin’ rope into neat lengths. "
            "Each piece gets an ID, and the model learns to predict the next ID. "
            "With BPE, those pieces are often parts of words."
        )
    elif concept == "perplexity":
        body = (
            "Perplexity measures how surprised the model is by the text. Lower perplexity means it predicts the next words more confidently; "
            "higher means it’s guessin’ more. It’s closely tied to loss."
        )
    elif concept == "checkpoint":
        body = (
            "A checkpoint is a saved snapshot of the model weights. If training stops, you can resume from that point instead of startin’ over. "
            "It also lets you keep the best model based on validation loss."
        )
    elif concept == "mixed precision (AMP)":
        body = (
            "AMP uses lower-precision math (like float16) for many operations to run faster and save memory. "
            "It keeps some parts in higher precision so trainin’ stays stable. "
            "Often ye get speed without losin’ accuracy."
        )
    else:
        body = (
            f"{concept.capitalize()} be a core idea in training models. "
            "In plain words, it helps the model learn patterns from data and make better predictions over time. "
            "Keep an eye on validation loss to make sure it generalizes."
        )

    return f"{opener} {body} {ender}"

def pirate_compare(a, b):
    opener = random.choice(OPENERS)
    ender = random.choice(CLOSERS)
    return (
        f"{opener} {a.capitalize()} is what ye see on the data the model trains on, while {b} is measured on held-out data. "
        f"If {a} drops but {b} rises, the model may be overfittin’—learnin’ the drills, not the real voyage. {ender}"
    )

def pirate_advice(topic):
    opener = random.choice(OPENERS)
    ender = random.choice(CLOSERS)

    # 3–6 bullet-like sentences, but still plain text
    if "focused" in topic:
        body = "Pick one task, set a 25-minute timer, and stow distractions out o’ sight. Take a 5-minute break, then sail again. End by writin’ the next step so ye start fast next time."
    elif "stress" in topic or "exam" in topic:
        body = "Breathe slow (in 4, hold 4, out 6), then do one quick review pass and stop. Sleep be part o’ the study, matey. On the exam, start with easy questions to build confidence."
    elif "emails" in topic:
        body = "Put the main point in the first line. Use short paragraphs and bullets for asks. End with a clear next step and deadline, then trim extra words like frayed rope."
    elif "presenting" in topic:
        body = "Practice yer opening twice, keep 3–5 key points, and speak slower than ye think. If a question hits, pause, repeat it, then answer with one clear idea first."
    elif "sleep" in topic:
        body = "Set a fixed wake time like a ship’s bell. Get morning light, stop caffeine after midday, and cut screens 30 minutes before bed. Write worries down so they don’t rattle in yer head."
    elif "routine" in topic:
        body = "Choose one anchor habit (same wake time), then add one small task after it. Keep goals tiny at first, track it daily, and adjust weekly like settin’ sail with the wind."
    elif "procrastinating" in topic:
        body = "Make the first step so small it feels silly—open the file, write one sentence. Work for 10 minutes only; once ye start, momentum follows. Reward yerself after each session."
    elif "feedback" in topic:
        body = "Start with what went well, then name one specific change with an example. Focus on the work, not the person, and agree on a next step. Keep it short and kind."
    else:
        body = "Keep it simple: pick one goal, break it into small steps, and do the first step now. Remove distractions and track progress daily. Small steady sails beat big storms."

    return f"{opener} {body} {ender}"

def make_example(kind, i):
    # Build a user prompt + assistant response
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
        # Pick relevant slots
        assistant = pirate_story(
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
        # Sometimes do compare question
        if random.random() < 0.25:
            a, b = random.choice(CONCEPT_PAIRS)
            user = f"What's the difference between {a} and {b}?"
            assistant = pirate_compare(a, b)
        else:
            concept = random.choice(CONCEPTS + ["overfitting", "tokenizer", "perplexity", "checkpoint", "mixed precision (AMP)"])
            template = random.choice(EXPLAIN_PROMPTS)
            user = template.format(concept=concept, concept_a=random.choice(CONCEPTS), concept_b=random.choice(CONCEPTS))
            assistant = pirate_explain(concept)
    else:  # advice
        user = random.choice(ADVICE_PROMPTS)
        assistant = pirate_advice(user)

    ex = {
        "conversation": [
            {"text": user},
            {"text": assistant},
        ]
    }
    return ex

def main():
    out_path = Path(__file__).resolve().parent / "pirate_persona.jsonl"

    n_total = 500
    n_story = n_total // 3
    n_explain = n_total // 3
    n_advice = n_total - n_story - n_explain

    kinds = (["story"] * n_story) + (["explain"] * n_explain) + (["advice"] * n_advice)
    random.shuffle(kinds)

    with out_path.open("w", encoding="utf-8") as f:
        for i, k in enumerate(kinds, start=1):
            ex = make_example(k, i)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {n_total} examples to: {out_path}")

if __name__ == "__main__":
    main()