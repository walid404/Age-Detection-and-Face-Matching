import matplotlib.pyplot as plt
import torch

def visualize_samples(model, dataset, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))

    for i in range(num_samples):
        img, age = dataset[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0)).item()

        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"T: {age:.1f} | P: {pred:.1f}")
        axes[i].axis("off")

    plt.show()
