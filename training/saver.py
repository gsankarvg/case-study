import json
import os

def save_model(model, path):
    model.save(path)

def save_metrics(metrics_dict, path):
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
