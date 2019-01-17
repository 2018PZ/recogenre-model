import matplotlib.pyplot as plt


# Plot and save keras trainning history
def save_history(hist, save_dir):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'], label='train')
    plt.plot(hist.history['val_acc'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir, format='png', bbox_inches='tight')
