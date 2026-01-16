import yaml
from controller.train_controller import run_training

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

run_training(config)
