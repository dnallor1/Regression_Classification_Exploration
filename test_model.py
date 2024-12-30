import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from joblib import load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import wandb

# Initialize WandB
wandb.init(project="regression_classification_exploration")

# Function to load the trained model
def load_model(task):
    filename = f"trained_model_{task}.joblib"
    try:
        model = load(filename)
        print(f"Loaded {task} model from '{filename}'.")
        return model
    except FileNotFoundError:
        print(f"Model file '{filename}' not found. Please train the model first.")
        exit()

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, task):
    y_pred = model.predict(X_test)

    # Calculate the median of X_test.shape
    x_test_shape_median = np.median(X_test.shape)

    if task == "classification":
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics to WandB
        wandb.log({
            "accuracy": accuracy,
            "f1_score": f1,
            "x_test_shape_median": x_test_shape_median  # Log the median
        })

        print(f"Classification Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Median of X_test.shape: {x_test_shape_median:.2f}")

    elif task == "regression":
        mse = mean_squared_error(y_test, y_pred)

        # Log metrics to WandB
        wandb.log({
            "mean_squared_error": mse,
            "x_test_shape_median": x_test_shape_median  # Log the median
        })

        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Median of X_test.shape: {x_test_shape_median:.2f}")

    return y_pred

# Visualization function for test data
def visualize_test_data(X_test, y_test, y_pred, task):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    plt.figure(figsize=(8, 6))
    if task == "classification":
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="viridis", s=10, label="Predictions")
        plt.colorbar(scatter, label="Predicted Classes")
    elif task == "regression":
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap="coolwarm", s=10, label="Predictions")
        plt.colorbar(scatter, label="Predicted Values")
    plt.title(f"PCA Visualization of {task.capitalize()} Predictions")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.show()

    # Log visualization to WandB
    wandb.log({"PCA Visualization": wandb.Image(plt)})

# Switch between tasks
print("Choose the task to test:")
print("1. Classification")
print("2. Regression")
choice = input("Enter 1 for Classification or 2 for Regression: ")

if choice == "1":
    task = "classification"
elif choice == "2":
    task = "regression"
else:
    print("Invalid choice. Exiting.")
    exit()

# Load test data
print(f"Loading {task} test data...")
if task == "classification":
    from sklearn.datasets import make_classification
    X_test, y_test = make_classification(n_samples=200, n_features=10, n_informative=8, n_classes=2, random_state=42)
elif task == "regression":
    from sklearn.datasets import make_regression
    X_test, y_test = make_regression(n_samples=200, n_features=10, noise=0.1, random_state=42)

# Load the trained model
model = load_model(task)

# Evaluate the model
print(f"Evaluating the {task} model...")
y_pred = evaluate_model(model, X_test, y_test, task)

# Visualize predictions
visualize_test_data(X_test, y_test, y_pred, task)

# Automated Testing Options
def automated_testing(model):
    print("\n--- Automated Testing Options ---")
    print("1. Test with random generated feature values")
    print("2. Test with a batch of feature samples")
    print("3. Exit")
    choice = input("Choose an option (1, 2, or 3): ")

    if choice == "1":
        # Test with a single random sample
        test_sample = np.random.uniform(-2, 2, X_test.shape[1]).reshape(1, -1)
        prediction = model.predict(test_sample)
        print(f"Random Test Sample: {test_sample}")
        print(f"Predicted: {prediction[0]}")

    elif choice == "2":
        # Test with a batch of samples
        batch_size = int(input("Enter the number of test samples to generate: "))
        test_samples = np.random.uniform(-2, 2, (batch_size, X_test.shape[1]))
        predictions = model.predict(test_samples)
        print(f"Generated Test Samples: \n{test_samples}")
        print(f"Predictions: {predictions}")

    elif choice == "3":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Exiting...")

# Uncomment the following line to enable automated testing
automated_testing(model)

# Finish WandB run
wandb.finish()
