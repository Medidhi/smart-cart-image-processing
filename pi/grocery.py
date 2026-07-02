"""
Grocery-store subset of COCO + display helpers.

Off-the-shelf YOLOv8 (COCO) only covers the produce / food / container classes
below. Broader coverage (potato, tomato, onion, peppers, leafy greens, packaged
goods, cans, boxes) requires a custom-trained model — see README Phase 2.
"""

# COCO labels that appear in a grocery store, grouped for reference.
PRODUCE = {"banana", "apple", "orange", "broccoli", "carrot"}
PREPARED_FOOD = {"sandwich", "hot dog", "pizza", "donut", "cake"}
CONTAINERS = {"bottle", "wine glass", "cup", "bowl"}
UTENSILS = {"fork", "knife", "spoon"}

GROCERY_CLASSES = PRODUCE | PREPARED_FOOD | CONTAINERS | UTENSILS

# Stable BGR color per category so the same item type is always the same color.
_CATEGORY_COLOR = {
    "produce": (0, 200, 0),        # green
    "prepared_food": (0, 165, 255),  # orange
    "containers": (255, 128, 0),   # blue
    "utensils": (200, 0, 200),     # magenta
}


def category_of(label):
    if label in PRODUCE:
        return "produce"
    if label in PREPARED_FOOD:
        return "prepared_food"
    if label in CONTAINERS:
        return "containers"
    if label in UTENSILS:
        return "utensils"
    return "other"


def is_grocery(label):
    return label in GROCERY_CLASSES


def color_for(label):
    return _CATEGORY_COLOR.get(category_of(label), (0, 255, 0))


def filter_grocery(dets):
    return [d for d in dets if is_grocery(d.label)]
