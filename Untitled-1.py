# %%
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class ElasticNetLoss(nn.Module):
    def __init__(self, model, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy_loss(outputs, targets)
        l1_norm = sum(param.abs().sum() for param in self.model.parameters())
        l2_norm = sum(param.pow(2).sum() for param in self.model.parameters())
        elastic_net_penalty = self.alpha * (
            self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm
        )
        return ce_loss + elastic_net_penalty

# %%
# Load the data
df1 = pd.read_excel(
    "/home/gddaslab/mxp140/sclerosis_project/miRNA_signal_hsa_number2.xlsx",
    engine="openpyxl",
    sheet_name="Sheet1",
)

# Drop non-feature columns
df = df1.drop(columns=["ID", "Transcript_ID"])
df = df.iloc[:, 10:]

# Label the columns based on their types
labels = {"aHC": 0, "sMS": 1, "aMS": 2, "aPOMS": 3, "sPOMS": 4, "pBar": 5}

# Create target labels for each column
y = []
for col in df.columns:
    for key in labels.keys():
        if col.startswith(key):
            y.append(labels[key])
            break

# %%
# Define the number of folds for the cross-validation
n_folds = 10
kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# %%
# Initialize variables to store the best model and its accuracy
best_model = None
best_accuracy = 0.0

# %%
# Convert DataFrame to tensor
X = df.T.values
y = y

# Convert the entire dataset to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# %%
# Cross-validation loop
for fold, (train_ids, val_ids) in enumerate(kfold.split(X_tensor)):
    print(f"Fold {fold+1}/{n_folds}")
    
    # Split the data into training and validation sets
    X_train, X_val = X_tensor[train_ids], X_tensor[val_ids]
    y_train, y_val = y_tensor[train_ids], y_tensor[val_ids]
    
    # Instantiate the model
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))
    model = SoftmaxRegression(input_dim, output_dim)
    
    # Define loss function and optimizer
    criterion = ElasticNetLoss(model, alpha=0.01, l1_ratio=0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    with torch.no_grad():
        model.eval()
        val_outputs = model(X_val)
        _, y_pred_tensor = torch.max(val_outputs, 1)
        y_pred = y_pred_tensor.numpy()
        y_true = y_val.numpy()

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Validation Accuracy: {accuracy:.2f}")

        # If this model is better than the previous best, update the best model and accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

# Save the best model
torch.save(best_model.state_dict(), "softmax_classifier_best.pth")
print(f"Best Validation Accuracy: {best_accuracy:.2f}")

# %%
# Save the best model
torch.save(best_model.state_dict(), "softmax_classifier_wo_pHC_best.pth")
print(f"Best Validation Accuracy: {best_accuracy:.2f}")