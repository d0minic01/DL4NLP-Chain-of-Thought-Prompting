#!/usr/bin/env python3
"""Generate synthetic test data for symbolic and commonsense reasoning benchmarks.

Outputs (all in data/):
  coin_flip_test.jsonl              500 coin-flip instances (2-4 people)
  last_letter_concatenation_test.jsonl  500 last-letter-concat instances (2-4 words)
  date_understanding_test.jsonl     ~240 date-arithmetic instances
  sports_understanding_test.jsonl   ~156 athlete plausibility instances
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)
Path("data").mkdir(exist_ok=True)


# ─── Coin Flip ─────────────────────────────────────────────────────────────────

NAMES = [
    "Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter",
    "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    "Yara", "Zoe", "Aaron", "Beth", "Caleb", "Diana", "Ethan", "Fiona",
    "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Marcus", "Nina",
    "Oscar", "Pam", "Ray", "Sofia", "Tom", "Ursula", "Vince", "Wanda",
    "Xavier", "Yvonne", "Zach", "Amber", "Blake", "Chloe", "Derek",
    "Emma", "Felix", "Gina", "Hugo", "Irene", "James", "Leah", "Miles",
]


def make_coin_flip(n_people=None):
    if n_people is None:
        n_people = random.randint(2, 4)
    people = random.sample(NAMES, n_people)
    flips = [random.choice([True, False]) for _ in range(n_people)]

    parts = []
    for name, flipped in zip(people, flips):
        if flipped:
            parts.append(f"{name} flips the coin.")
        else:
            parts.append(f"{name} does not flip the coin.")

    question = "A coin is heads up. " + " ".join(parts) + " Is the coin still heads up?"
    answer = "yes" if sum(flips) % 2 == 0 else "no"
    return {"question": question, "target": answer}


coin_flip_data = [make_coin_flip() for _ in range(500)]
with open("data/coin_flip_test.jsonl", "w") as f:
    for item in coin_flip_data:
        f.write(json.dumps(item) + "\n")
print(f"coin_flip:               {len(coin_flip_data)} examples")


# ─── Last Letter Concatenation ─────────────────────────────────────────────────

WORDS = [
    "apple", "bridge", "cloud", "dragon", "eagle", "forest", "garden",
    "harbor", "island", "jungle", "knight", "lemon", "mango", "novel",
    "orange", "planet", "queen", "river", "stone", "tiger", "uncle",
    "valley", "window", "yellow", "zebra", "anchor", "bottle", "castle",
    "desert", "engine", "flame", "guitar", "honey", "jacket", "kettle",
    "lantern", "mirror", "needle", "oven", "puzzle", "quilt", "rocket",
    "silver", "table", "umbrella", "violet", "walnut", "xenon", "yarn",
    "butter", "candle", "diamond", "elbow", "feather", "glove", "hammer",
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Liam", "Mary", "Nathan", "Olivia", "Paul",
    "Quinn", "Ruth", "Scott", "Tara", "Ulric", "Vera", "Walter", "Xena",
    "Yasmin", "Zach", "Brian", "Clara", "Donna", "Elliot",
]


def make_last_letter(n_words=None):
    if n_words is None:
        n_words = random.randint(2, 4)
    words = random.sample(WORDS, n_words)
    phrase = " ".join(words)
    answer = "".join(w[-1].lower() for w in words)
    question = f'Take the last letters of the words in "{phrase}" and concatenate them.'
    return {"question": question, "target": answer}


llc_data = [make_last_letter() for _ in range(500)]
with open("data/last_letter_concatenation_test.jsonl", "w") as f:
    for item in llc_data:
        f.write(json.dumps(item) + "\n")
print(f"last_letter_concat:      {len(llc_data)} examples")


# ─── Date Understanding ────────────────────────────────────────────────────────

BASE_DATES = [
    date(1990, 3, 15), date(2001, 7, 22), date(1985, 11, 4),
    date(2010, 1, 30), date(1975, 6, 12), date(2005, 9, 8),
    date(1998, 12, 25), date(2015, 4, 3), date(1963, 8, 19),
    date(2020, 2, 14), date(1980, 10, 31), date(2008, 5, 17),
    date(1955, 3, 7),  date(2018, 11, 28), date(1970, 7, 4),
    date(2012, 6, 21), date(1992, 2, 28),  date(2003, 9, 1),
    date(1967, 12, 30), date(2016, 1, 15),
]

DELTAS = [1, 3, 7, 10, 14, 30]


def fmt(d: date) -> str:
    return d.strftime("%m/%d/%Y")


date_data = []
for base in BASE_DATES:
    for delta in DELTAS:
        future = base + timedelta(days=delta)
        past = base - timedelta(days=delta)
        date_data.append({
            "question": f"Today is {fmt(base)}. What is the date {delta} days from now in MM/DD/YYYY?",
            "target": fmt(future),
        })
        date_data.append({
            "question": f"Today is {fmt(base)}. What is the date {delta} days ago in MM/DD/YYYY?",
            "target": fmt(past),
        })

# Phrasing variants for week-based offsets
for base in BASE_DATES[:10]:
    future = base + timedelta(weeks=1)
    past = base - timedelta(weeks=1)
    date_data.append({
        "question": f"Today is {fmt(base)}. What is the date one week from today in MM/DD/YYYY?",
        "target": fmt(future),
    })
    date_data.append({
        "question": f"Today is {fmt(base)}. What is the date one week ago in MM/DD/YYYY?",
        "target": fmt(past),
    })

# Paper-style examples (kept from original exemplars to avoid train/test overlap)
date_data += [
    {"question": "It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?",
     "target": "04/20/1969"},
    {"question": "Today is 03/05/2023. What is the date 30 days from now in MM/DD/YYYY?",
     "target": "04/04/2023"},
    {"question": "Today is 12/28/2021. What is the date 7 days from now in MM/DD/YYYY?",
     "target": "01/04/2022"},
    {"question": "Today is 01/01/2000. What is the date 1 day ago in MM/DD/YYYY?",
     "target": "12/31/1999"},
]

random.shuffle(date_data)
with open("data/date_understanding_test.jsonl", "w") as f:
    for item in date_data:
        f.write(json.dumps(item) + "\n")
print(f"date_understanding:      {len(date_data)} examples")


# ─── Sports Understanding ──────────────────────────────────────────────────────

ATHLETES: dict[str, str] = {
    "LeBron James": "basketball",
    "Stephen Curry": "basketball",
    "Kevin Durant": "basketball",
    "Kobe Bryant": "basketball",
    "Giannis Antetokounmpo": "basketball",
    "James Harden": "basketball",
    "Tom Brady": "american football",
    "Aaron Rodgers": "american football",
    "Patrick Mahomes": "american football",
    "Peyton Manning": "american football",
    "Jerry Rice": "american football",
    "Wayne Gretzky": "hockey",
    "Sidney Crosby": "hockey",
    "Alex Ovechkin": "hockey",
    "Connor McDavid": "hockey",
    "Patrick Kane": "hockey",
    "Lionel Messi": "soccer",
    "Cristiano Ronaldo": "soccer",
    "Neymar": "soccer",
    "Kylian Mbappe": "soccer",
    "Raheem Sterling": "soccer",
    "Mike Trout": "baseball",
    "Derek Jeter": "baseball",
    "Babe Ruth": "baseball",
    "Clayton Kershaw": "baseball",
    "David Ortiz": "baseball",
}

SPORT_ACTIONS: dict[str, list[str]] = {
    "basketball": [
        "dunked over the defender",
        "hit a buzzer-beating three-pointer",
        "set the pick and roll",
        "made the free throw",
        "blocked the shot",
        "banked the shot in",
        "made the alley-oop",
        "grabbed the rebound",
        "stole the ball",
        "made a behind-the-back pass",
    ],
    "american football": [
        "threw a touchdown pass",
        "caught the screen pass",
        "sacked the quarterback",
        "kicked the field goal",
        "tackled the running back",
        "scored a rushing touchdown",
        "intercepted the pass",
        "returned the kickoff for a touchdown",
        "completed a Hail Mary",
        "rushed for a first down",
    ],
    "hockey": [
        "was called for slashing",
        "scored a hat trick",
        "iced the puck",
        "was checked into the boards",
        "scored on a wrist shot",
        "made a breakaway goal",
        "was penalized for hooking",
        "won the face-off",
        "scored a power play goal",
        "made a glove save",
    ],
    "soccer": [
        "scored a bicycle kick goal",
        "made a sliding tackle",
        "bent the free kick into the net",
        "was booked with a yellow card",
        "scored a penalty kick",
        "made a diving header",
        "completed a hat trick",
        "dribbled past the goalkeeper",
        "was called for offside",
        "made a long-range strike",
    ],
    "baseball": [
        "hit a walk-off home run",
        "struck out the side",
        "stole second base",
        "hit a grand slam",
        "made a diving catch in center field",
        "threw a complete game shutout",
        "hit a double off the wall",
        "turned a double play",
        "hit a sacrifice fly",
        "walked with the bases loaded",
    ],
}

sports_data = []
all_sports = list(SPORT_ACTIONS.keys())

for athlete, athlete_sport in ATHLETES.items():
    for action in SPORT_ACTIONS[athlete_sport][:4]:
        q = f'Is the following sentence plausible? "{athlete} {action}."'
        sports_data.append({"question": q, "target": "yes"})

    other_sports = [s for s in all_sports if s != athlete_sport]
    for other_sport in random.sample(other_sports, 2):
        action = random.choice(SPORT_ACTIONS[other_sport])
        q = f'Is the following sentence plausible? "{athlete} {action}."'
        sports_data.append({"question": q, "target": "no"})

random.shuffle(sports_data)
with open("data/sports_understanding_test.jsonl", "w") as f:
    for item in sports_data:
        f.write(json.dumps(item) + "\n")
print(f"sports_understanding:    {len(sports_data)} examples")
