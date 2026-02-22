"""
Object bank for Seekr.
Each entry has:
  displayName  - shown to players
  prompts      - positive CLIP descriptions
  negatives    - things to distinguish from (combined with GLOBAL_NULLS)
  threshold    - min positive similarity to count as detected
  margin       - min gap between positive and negative score
"""

OBJECTS: dict = {
    "water_bottle": {
        "displayName": "WATER BOTTLE",
        "prompts": [
            "a water bottle",
            "a plastic water bottle",
            "a reusable water bottle",
            "someone holding a water bottle",
        ],
        "negatives": ["a coffee mug", "a glass", "a vase", "a phone", "a spoon"],
        "threshold": 0.22,
        "margin": 0.04,
    },
    "phone": {
        "displayName": "SMARTPHONE",
        "prompts": [
            "a smartphone",
            "a mobile phone",
            "someone holding a phone",
            "a phone with a screen",
        ],
        "negatives": ["a tablet", "a remote control", "a calculator", "a water bottle", "a keyboard"],
        "threshold": 0.22,
        "margin": 0.04,
    },
    "notebook": {
        "displayName": "NOTEBOOK",
        "prompts": [
            "a spiral notebook",
            "someone holding a notebook",
            "a lined writing notebook",
            "a notebook on a desk",
        ],
        "negatives": ["a book", "a tablet", "a sketchbook", "a magazine", "a keyboard"],
        "threshold": 0.22,
        "margin": 0.04,
    },
    "keyboard": {
        "displayName": "KEYBOARD",
        "prompts": [
            "a computer keyboard",
            "a mechanical keyboard",
            "a USB keyboard",
            "someone holding a keyboard",
            "a keyboard on a desk",
        ],
        "negatives": ["a remote control", "a piano", "a calculator", "a phone", "a laptop"],
        "threshold": 0.22,
        "margin": 0.04,
    },
    "charger": {
        "displayName": "PHONE CHARGER",
        "prompts": [
            "a phone charger cable",
            "a USB charging cable",
            "someone holding a charger",
            "a power cable",
        ],
        "negatives": ["earphones", "a wire", "a rope", "a belt"],
        "threshold": 0.22,
        "margin": 0.04,
    },
    "spoon": {
        "displayName": "SPOON",
        "prompts": [
            "a spoon",
            "a metal spoon",
            "someone holding a spoon",
            "a spoon on a table",
        ],
        "negatives": ["a fork", "a knife", "chopsticks", "a ladle", "a water bottle"],
        "threshold": 0.22,
        "margin": 0.04,
    },
}

GLOBAL_NULLS = [
    "background clutter",
    "a table surface",
    "an empty room",
    "a wall",
    "the floor",
]


def get_object(object_id: str) -> dict:
    return OBJECTS[object_id]


def get_all_ids() -> list[str]:
    return list(OBJECTS.keys())
