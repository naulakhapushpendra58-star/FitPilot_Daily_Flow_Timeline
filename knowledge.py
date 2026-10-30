import json
from pathlib import Path

KB_PATH = Path(__file__).with_name("knowledge_base.json")


def load_knowledge_base(path: Path = KB_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_ascii(value: str) -> str:
    return value.encode("ascii", "ignore").decode("ascii")
