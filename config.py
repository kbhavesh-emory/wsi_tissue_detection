# Map DSA annotation labels to index values. Background index is 0.
label2idx = {
    "Background": 0,
    "Gray Matter": 1,
    "White Matter": 2,
    "Superficial": 3,
    "Leptomeninges": 4,
    "Exclude": 5,
}

tissue_label2id = {"htk-manual-tissue": 1}
tissue_train_label2id = {"background": 0, "tissue": 1}

COLORS = {
    0: {
        "fillColor": "rgba(255, 255, 255, 0.5)",
        "lineColor": "rgb(255, 255, 255)",
    },
    1: {"fillColor": "rgba(0, 128, 0, 0.5)", "lineColor": "rgb(0, 128, 0)"},
    2: {"fillColor": "rgba(0, 0, 255, 0.5)", "lineColor": "rgb(0, 0, 255)"},
    3: {
        "fillColor": "rgba(255, 255, 0, 0.5)",
        "lineColor": "rgb(255, 255, 0)",
    },
    4: {"fillColor": "rgba(0, 0, 0, 0.5)", "lineColor": "rgb(0, 0, 0)"},
    5: {"fillColor": "rgba(255, 0, 0, 0.5)", "lineColor": "rgb(255, 0, 0)"},
}
