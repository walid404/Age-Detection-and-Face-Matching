import os
import matplotlib.pyplot as plt


def plot_age_gaps(gaps, split_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.hist(gaps, bins=20)
    plt.xlabel("Age Gap (Years)")
    plt.ylabel("Number of Identities")
    plt.title(f"Cross-Age Gap – {split_name}")
    plt.savefig(os.path.join(save_dir, f"{split_name}_age_gaps.png"))
    plt.close()
