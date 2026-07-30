from pathlib import Path

import yaml


config_path = Path("configs/damage_detection.yaml")

with config_path.open("r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

print("Configuration loaded successfully.")
print(config)