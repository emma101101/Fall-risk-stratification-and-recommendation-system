import json
import re
from pathlib import Path

FRIDS_PATH = Path("frids_library.json")

with FRIDS_PATH.open("r", encoding="utf-8") as f:
    FRIDS_LIBRARY = json.load(f)

FRIDS_MEDICATION_NAMES = FRIDS_LIBRARY["medication_names"]

FRIDS_PATTERNS = [
    rf"\b{re.escape(med.lower())}\b"
    for med in FRIDS_MEDICATION_NAMES
]

def has_frids_medication(text: str) -> bool:
    t = text.lower()
    return any(re.search(pattern, t) for pattern in FRIDS_PATTERNS)
