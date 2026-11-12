"""PrivAI - Day 4: Model router.

Looks at the question and decides WHICH local model should answer it.
Step 1: fast keyword rules (free, instant).
Step 2: if rules can't decide, ask a small model to classify (slower, smarter).
Every decision is saved in routing_log.csv - useful later to measure accuracy.
"""
import csv
import re
import time
from pathlib import Path

import ollama

# Which models are best for which task (first available one is used).
# Change these to the models you actually have in `ollama list`.
MODEL_MAP = {
    "coding":   ["qwen2.5-coder:7b", "qwen2.5:7b", "llama3.2:3b"],
    "document": ["qwen2.5:7b", "llama3.2:3b"],   # follows "use only context" better
    "summary":  ["llama3.2:3b", "qwen2.5:7b"],   # small + fast is enough
    "general":  ["llama3.2:3b", "qwen2.5:7b"],
}

CODE_RE = re.compile(
    r"```|\b(code|python|java|javascript|c\+\+|sql|html|css|function|class|"
    r"bug|debug|error|exception|compile|script|regex|algorithm|api|stack trace)\b",
    re.I,
)
SUMMARY_RE = re.compile(
    r"\b(summar\w*|tl;?dr|key points|shorten|paraphrase|rewrite|in short|brief)\b",
    re.I,
)
LOG_FILE = Path("routing_log.csv")


def pick_model(task, available):
    """Choose the best installed model for the task (in order of preference)."""
    for p in MODEL_MAP[task]:
        if p in available:                # exact match, e.g. qwen2.5:7b
            return p
        base = p.split(":")[0]
        for a in available:               # same family, any size (qwen2.5:3b)
            if a.startswith(base):
                return a
    return available[0]                   # last resort


def classify_with_llm(question, available):
    """Ask a small model to label the question. Returns a task name."""
    model = pick_model("general", available)
    prompt = (
        "Classify the user message into exactly one word: "
        "coding, summary, or general.\n"
        f"Message: {question}\nAnswer:"
    )
    try:
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 5},
        )
        label = res["message"]["content"].strip().lower()
        for task in ("coding", "summary"):
            if task in label:
                return task
    except Exception:
        pass
    return "general"


def route(question, use_docs, available):
    """Return (task, model, reason)."""
    if use_docs:
        task, reason = "document", "RAG is on, so it's a document question"
    elif CODE_RE.search(question):
        task, reason = "coding", "matched a coding keyword"
    elif SUMMARY_RE.search(question):
        task, reason = "summary", "matched a summary keyword"
    else:
        task = classify_with_llm(question, available)
        reason = "no rule matched, small model classified it"

    model = pick_model(task, available)
    log_decision(question, task, model, reason)
    return task, model, reason


def log_decision(question, task, model, reason):
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "question", "task", "model", "reason"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), question[:80],
                    task, model, reason])