import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim

# 1. SETUP: Replace with your DagsHub or Remote MLflow URI
# If running locally for a test, comment these two lines out.
mlflow.set_tracking_uri("https://dagshub.com/YourUsername/YourRepo.mlflow")
mlflow.set_experiment("Cheap-GPU-Analysis")

# Simple Model
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 2. START RUN
with mlflow.start_run(run_name="Initial_GPU_Test"):
    # Log hyperparameters
    params = {"lr": 0.01, "epochs": 10, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    mlflow.log_params(params)

    # Dummy Training Loop
    for epoch in range(params["epochs"]):
        inputs = torch.randn(64, 10)
        targets = torch.randn(64, 1)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 3. LOG METRICS (Check these in real-time on your dashboard)
        mlflow.log_metric("loss", loss.item(), step=epoch)
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # 4. LOG ARTIFACTS (The Model)
    # Using 'log_model' allows for easy deployment later
    mlflow.pytorch.log_model(model, "model_artifacts")
    
    print("Run complete. Check your remote MLflow UI!")