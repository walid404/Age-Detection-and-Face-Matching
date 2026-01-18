import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses,
                plots_dir: str = "src/reports/plots", 
                loss_name: str = "Loss"):
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(loss_name)
    plt.title(f"{loss_name} over Epochs")
    plt.savefig(f"{plots_dir}/{loss_name.lower()}_over_epochs.png")
    plt.close()
