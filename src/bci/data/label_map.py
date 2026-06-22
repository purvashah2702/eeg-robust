"""Common label map across datasets — unifies different naming conventions."""

# Common target labels
COMMON_LABELS = {
    "left_hand": 0,
    "right_hand": 1,
    "feet": 2,
    "rest": 3,
    "tongue": 4,
}

# Maps each dataset's raw event names -> common label names
DATASET_LABEL_MAPS = {
    "bnci2014_001": {
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "feet": "feet",
        "tongue": "tongue",
    },
    "bnci2014_002": {
        "right_hand": "right_hand",
        "feet": "feet",
    },
    "physionet": {
        "left_hand": "left_hand",
        "right_hand": "right_hand",
        "hands": "rest",
        "feet": "feet",
        "rest": "rest",
    },
}


def remap_event_id(event_id: dict, dataset_name: str) -> dict:
    """Convert a dataset's raw event_id dict to common label integers."""
    label_map = DATASET_LABEL_MAPS[dataset_name]
    new_event_id = {}
    for raw_label, code in event_id.items():
        if raw_label in label_map:
            common_label = label_map[raw_label]
            new_event_id[common_label] = COMMON_LABELS[common_label]
    return new_event_id
