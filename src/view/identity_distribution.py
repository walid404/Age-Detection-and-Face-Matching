import os
import matplotlib.pyplot as plt
from collections import Counter


def plot_identity_distribution(subsets: list, names: list, plots_dir: str = "src/reports/plots"):
    os.makedirs(plots_dir, exist_ok=True)

    for subset, name in zip(subsets, names):
        person_ids = subset.dataset.data.iloc[subset.indices]["person_id"]
        counts = Counter(person_ids)

        plt.figure()
        plt.hist(list(counts.values()), bins=20)
        plt.xlabel("Images per Person")
        plt.ylabel("Frequency")
        plt.title(f"Identity Distribution – {name}")
        plt.savefig(os.path.join(plots_dir, f"{name}_identity_distribution.png"))
        plt.close()
