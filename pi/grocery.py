"""
Grocery classes + display helpers, for BOTH detection models:

- Off-the-shelf COCO YOLOv8: only the produce / food / container subset below
  appears in a grocery store; everything else is filtered out.
- Custom 43-class grocery model (training/data.yaml -> grocery_labels.py):
  every class IS a grocery item, so nothing is filtered; categories follow
  the data.yaml grouping (fruit / dairy / vegetables).
"""
from grocery_labels import GROCERY_NAMES

# COCO labels that appear in a grocery store, grouped for reference.
PRODUCE = {"banana", "apple", "orange", "broccoli", "carrot"}
PREPARED_FOOD = {"sandwich", "hot dog", "pizza", "donut", "cake"}
CONTAINERS = {"bottle", "wine glass", "cup", "bowl"}
UTENSILS = {"fork", "knife", "spoon"}

GROCERY_CLASSES = PRODUCE | PREPARED_FOOD | CONTAINERS | UTENSILS

# Custom grocery model classes, grouped by their data.yaml index ranges.
GROCERY_FRUIT = set(GROCERY_NAMES[0:19])
GROCERY_DAIRY = set(GROCERY_NAMES[19:28])
GROCERY_VEG = set(GROCERY_NAMES[28:43])
GROCERY_MODEL_CLASSES = GROCERY_FRUIT | GROCERY_DAIRY | GROCERY_VEG

# Stable BGR color per category so the same item type is always the same color.
_CATEGORY_COLOR = {
    "produce": (0, 200, 0),        # green
    "prepared_food": (0, 165, 255),  # orange
    "containers": (255, 128, 0),   # blue
    "utensils": (200, 0, 200),     # magenta
    "fruit": (0, 200, 0),          # green
    "dairy": (255, 220, 120),      # light blue
    "vegetable": (60, 160, 0),     # dark green
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
    if label in GROCERY_FRUIT:
        return "fruit"
    if label in GROCERY_DAIRY:
        return "dairy"
    if label in GROCERY_VEG:
        return "vegetable"
    return "other"


def is_grocery(label):
    return label in GROCERY_CLASSES or label in GROCERY_MODEL_CLASSES


def color_for(label):
    return _CATEGORY_COLOR.get(category_of(label), (0, 255, 0))


def filter_grocery(dets):
    return [d for d in dets if is_grocery(d.label)]
