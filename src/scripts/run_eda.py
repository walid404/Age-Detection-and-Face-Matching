import os
import pandas as pd
import matplotlib.pyplot as plt

EDA_DIR = "eda/eda_plots"
os.makedirs(EDA_DIR, exist_ok=True)

df = pd.read_csv("Dataset/FGNET/labels.csv")

# Age distribution
plt.figure()
plt.hist(df["age"], bins=30)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.savefig(f"{EDA_DIR}/age_distribution.png")
plt.close()

# Images per person
counts = df["person_id"].value_counts()

plt.figure()
plt.hist(counts, bins=30)
plt.xlabel("Images per Person")
plt.ylabel("Frequency")
plt.title("Images per Person Distribution")
plt.savefig(f"{EDA_DIR}/images_per_person.png")
plt.close()

# Age vs frequency
plt.figure()
df["age"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Frequency")
plt.savefig(f"{EDA_DIR}/age_frequency.png")
plt.close()
