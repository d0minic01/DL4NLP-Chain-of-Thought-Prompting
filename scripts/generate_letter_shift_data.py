#!/usr/bin/env python3
"""Generate synthetic test data for the letter-shift benchmark.

Task: A letter starts at a given position in the alphabet. Each person in a
sequence either "shifts" it forward by one position (wrapping A→Z) or
"keeps" it. What is the letter after all operations?

This extends the coin-flip symbolic task from binary state (heads/tails) to a
26-state modular counter, requiring the model to track a richer state through
a chain of operations.
"""

import json
import random
from pathlib import Path

random.seed(42)
Path("data").mkdir(exist_ok=True)

NAMES = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Beth", "Caleb", "Diana", "Ethan", "Fiona",
    "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina",
    "Oscar", "Pam", "Ray", "Sofia", "Tom", "Ursula", "Vince", "Wanda",
]

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def letter_index(c: str) -> int:
    return ALPHABET.index(c.lower())


def index_to_letter(i: int) -> str:
    return ALPHABET[i % 26]


def make_letter_shift(n_people=None):
    if n_people is None:
        n_people = random.randint(2, 4)

    start_letter = random.choice(ALPHABET)
    people = random.sample(NAMES, n_people)
    shifts = [random.choice([True, False]) for _ in range(n_people)]

    parts = []
    for name, shifted in zip(people, shifts):
        if shifted:
            parts.append(f"{name} shifts the letter.")
        else:
            parts.append(f"{name} does not shift the letter.")

    question = (
        f'The letter starts as "{start_letter}". '
        + " ".join(parts)
        + " What is the letter now?"
    )

    total_shifts = sum(shifts)
    final_idx = (letter_index(start_letter) + total_shifts) % 26
    answer = index_to_letter(final_idx)
    return {"question": question, "target": answer}


data = [make_letter_shift() for _ in range(500)]
with open("data/letter_shift_test.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
print(f"letter_shift: {len(data)} examples")
