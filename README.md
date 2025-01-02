# Regression_Classification_Exploration
A Python project for exploring regression and classification tasks using neural networks with metrics tracking via WandB.

## WandB Project and Run Details

This project uses [Weights and Biases (WandB)](https://wandb.ai) for tracking experiments and logging metrics.

- **Project Dashboard**: [View the WandB project here](https://wandb.ai/theodorerolland-poznan-university-of-technology)
- **API Key for Login**:  34dcabe3be76bcc128386ecc33cd934d9930c544
- **Project Overview**: [View the WandB project dashboard](https://wandb.ai/theodorerolland-poznan-university-of-technology/regression_classification_exploration?nw=nwusertheodorerolland)
- **Specific Run Details**: [View a sample run here](https://wandb.ai/theodorerolland-poznan-university-of-technology/regression_classification_exploration/runs/c4p8dkzx?nw=nwusertheodorerolland)

WandB is used to log:
- Model performance metrics such as accuracy, F1 score, or mean squared error.
- PCA visualizations of predictions on test data.
- Training and test dataset statistics.

### Generating the Trained Model
To train the model and generate the `.joblib` file:

1. Run the `train_model.py` script:
   ```bash
   python train_model.py
