import os
import pandas as pd
import matplotlib.pyplot as plt

def eda_plots(plot_dir: str = "src/reports/plots",
              dataset_root: str = "src/Dataset",
              dataset_name: str = "FGNET",
              labels_csv_name: str = "labels.csv"):
    
    os.makedirs(plot_dir, exist_ok=True)
    csv_path = os.path.join(dataset_root, dataset_name, labels_csv_name)
    df = pd.read_csv(csv_path)

    # Age distribution
    plt.figure()
    plt.hist(df["age"], bins=30)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution")
    plt.savefig(f"{plot_dir}/age_distribution.png")
    plt.close()

    # Images per person
    counts = df["person_id"].value_counts()

    plt.figure()
    plt.hist(counts, bins=30)
    plt.xlabel("Images per Person")
    plt.ylabel("Frequency")
    plt.title("Images per Person Distribution")
    plt.savefig(f"{plot_dir}/images_per_person.png")
    plt.close()

    # Age vs frequency
    plt.figure()
    df["age"].value_counts().sort_index().plot(kind="bar")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.title("Age Frequency")
    plt.savefig(f"{plot_dir}/age_frequency.png")
    plt.close()
    print(f"EDA plots saved to {plot_dir}")