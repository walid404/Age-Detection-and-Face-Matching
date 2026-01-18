import matplotlib.pyplot as plt
from src.controller.age_inference_controller import predict_age

def generate_age_prediction_samples(model, dataset, num_samples=5, plots_dir: str = "src/reports/plots"):
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img, age = dataset[i]
        pred = predict_age(model, img)

        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f"T: {age:.1f} | P: {pred:.1f}")
        axes[i].axis("off")

    plt.savefig(f"{plots_dir}/age_prediction_samples.png")
    plt.close()
