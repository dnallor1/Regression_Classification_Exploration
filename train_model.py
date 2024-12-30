import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
import wandb
from joblib import dump

# Initialize WandB
wandb.init(project="regression_classification_exploration", config={
    "test_size": 0.2,
    "random_state": 42,
    "n_features": 10,
    "n_classes": 2  # Only applicable for classification
})

# Ask user to choose task
print("Choose the task to perform:")
print("1. Classification")
print("2. Regression")
choice = input("Enter 1 for Classification or 2 for Regression: ").strip()

if choice == "1":
    task = "classification"
elif choice == "2":
    task = "regression"
else:
    print("Invalid choice. Exiting...")
    exit()

# Update the task in WandB config
wandb.config.update({"task": task})

# Configuration parameters
config = wandb.config

# Sample Data Generation
def generate_data(task, n_samples=1000, n_features=10, n_classes=2, random_state=42):
    if task == "classification":
        return make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_classes=n_classes,
            random_state=random_state
        )
    elif task == "regression":
        return make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=random_state
        )
    else:
        raise ValueError("Invalid task. Choose 'classification' or 'regression'.")

# Generate data
X, y = generate_data(
    task=config.task,
    n_samples=1000,
    n_features=config.n_features,
    n_classes=config.get("n_classes", 2),
    random_state=config.random_state
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.test_size, random_state=config.random_state)

# Update dataset details in WandB
wandb.config.update({
    "train_samples": X_train.shape[0],
    "test_samples": X_test.shape[0],
    "features": X_train.shape[1],
})

# Neural Network Model Construction
if config.task == "classification":
    model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=config.random_state)
elif config.task == "regression":
    model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=config.random_state)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
if config.task == "classification":
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics to WandB
    wandb.log({
        "accuracy": accuracy,
        "f1_score": f1
    })

    print(f"Classification Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    dump(model, "trained_model_Classification.joblib")
    print("Classification Training completed and model artifacts saved successfully.")

elif config.task == "regression":
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Log metrics to WandB
    wandb.log({
        "mean_squared_error": mse
    })

    print(f"Mean Squared Error: {mse:.2f}")
    dump(model, "trained_model_regression.joblib")
    print("Regression Training completed and model artifacts saved successfully.")

# Final Script for User Input/Output
def predict_user_input(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    if config.task == "classification":
        return f"Predicted Class: {int(prediction[0])}"
    elif config.task == "regression":
        return f"Predicted Value: {prediction[0]:.2f}"


# Example usage
example_input = X_test[0]  # Replace with real user input in practice
result = predict_user_input(example_input)
print(result)

# Finish WandB run
wandb.finish()
