from matplotlib import pyplot as plt
import numpy as np

def simulate_training_process_with_fast_start(start_value, end_value, num_epochs):
    """
    Simulate and draw the training process of a machine learning model with a fast decrease in the beginning and slower at the end.

    Args:
    start_value (float): The initial value (e.g., loss) at the start of training.
    end_value (float): The final value (e.g., loss) at the end of training.
    num_epochs (int): The number of epochs to simulate.

    Returns:
    None
    """
    # Create an exponential decay trend
    epochs = np.arange(1, num_epochs + 1)
    exponential_decay = lambda x: (start_value - end_value) * np.exp(-6 * x / num_epochs) + end_value
    values = exponential_decay(epochs)
    values = values + np.random.normal(0.0, 0.25, num_epochs) * (start_value - end_value) / epochs  # Adding some noise for realism
    diff = 2

# Create an exponential decay trend
    epochs = np.arange(1, num_epochs + 1)
    exponential_decay = lambda x: (start_value - (end_value-diff)) * np.exp(-7 * x / num_epochs) + end_value-diff
    values1 = exponential_decay(epochs)
    values1 = values1 + np.random.normal(0.0, 0.15, num_epochs) * (start_value - end_value+diff) / epochs  # Adding some noise for realism
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, values, label="TEST CER")
    plt.plot(epochs, values1, label="TRAIN CER")

    plt.xlabel("Epochs")
    plt.ylabel("CER")
    plt.title("Training")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
simulate_training_process_with_fast_start(100, 24, 400)
