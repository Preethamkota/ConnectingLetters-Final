from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["project_root"] = Path(".").resolve()
    config["data_root"] = Path(config["data_root"])
    return config
