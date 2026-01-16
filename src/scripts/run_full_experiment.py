import os
import yaml
from controller.train_controller import run_training
from scripts.generate_comparison_tables import generate_tables


def main():

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run_training(config)
    generate_tables()


if __name__ == "__main__":
    main()
