import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

# 0) prepare the data
df = pd.read_excel(
    "/home/gddaslab/mxp140/sclerosis_project/miRNA_signal_hsa_number2.xlsx",
    engine="openpyxl",
    sheet_name="Sheet1",
)
# print(df.columns)

# select all rows and only the columns whose names contain either 'sMS' or 'aMS'
data = df.loc[:, df.columns.str.contains('sPOMS') | df.columns.str.contains('aPOMS')]
print(data.head(5))

# Transpose data so that data is in shape of n_samples x n_features
data = data.transpose()

# Label sMS as 0 and aMS as 1
X, y = data.values, [0 if 'sPOMS' in idx else 1 for idx in data.index]

n_samples, n_features = X.shape

# split data in train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale data (recommended)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# print(type(X_train), type(X_test))

# convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

# reshape y_train as column vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = np.array(y_test).reshape(len(y_test), 1)
# print(y_train, y_test)

# 1) model
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class ElasticNetLoss(nn.Module):
    def __init__(self, model, alpha=1.0, l1_ratio=0.5):
        super(ElasticNetLoss, self).__init__()
        self.model = model
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.bceloss = nn.BCELoss()

    def forward(self, outputs, targets):
        bce_loss = self.bceloss(outputs, targets)
        l1_norm = sum(param.abs().sum() for param in self.model.parameters())
        l2_norm = sum(param.pow(2).sum() for param in self.model.parameters())
        elastic_net_penalty = self.alpha * (
            self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm
        )
        return bce_loss + elastic_net_penalty

model = LogisticRegression(n_features).to(device)
# print(model)

# 2) loss and optimizer
criterion = ElasticNetLoss(model, alpha=0.01, l1_ratio=0.5)
learning_rate = 0.01
optimizer = optim.SGD(model.parameters(), lr=learning_rate,)

# 3) Training loop
num_epochs = 1000
for epoch in range(num_epochs):

    # predict and calculate loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # Empty stale gradients, calculate gradients with backward pass, update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # print training results in each epoch
    if (epoch + 1) % 10 == 0:
        print(f"|| epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} ||")

# Evaluate the model
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_class = y_predicted.round()
    accuracy = accuracy_score(y_test, y_predicted_class.detach().cpu().numpy())
    print(f"\nAccuracy:{accuracy:.2f}")

# Save the model
model_filename = "softmax_classifier_stable_vs_active_peds.pth"
torch.save(model.state_dict(), model_filename)
print(f"Model saved as {model_filename}.")

# Note:
# In a binary logistic regression model, the output layer consists of a single neuron that predicts the probability of the positive class (class 1). 
# Therefore, the weight matrix of the linear layer (self.linear) has dimensions corresponding to the number of input features and a single output (i.e., n_features x 1). 
# This is why you see a one-dimensional tensor of weights.
# To identify the top 5 features contributing to each class, you can analyze the weights of the model. 
# The features with the highest positive weights contribute most to class 1, while those with the highest negative weights contribute most to class 0.

# Extract weights from the model
weights = model.linear.weight.data.cpu().numpy().flatten()

# Get the indices of the top 5 features for class 1 (highest positive weights)
top_5_class_1 = np.argsort(weights)[-5:]

# Get the indices of the top 5 features for class 0 (highest negative weights)
top_5_class_0 = np.argsort(weights)[:5]

print("Top 5 features contributing to class 1:", top_5_class_1)
print("Top 5 features contributing to class 0:", top_5_class_0)

# Optionally, you can print the feature names if you have them
feature_names = df['Transcript_ID']
print("Top 5 features contributing to class 1:", feature_names[top_5_class_1].values)
print("Top 5 features contributing to class 0:", feature_names[top_5_class_0].values)

table = pd.DataFrame({"sPOMS":feature_names[top_5_class_0].values,"aPOMS":feature_names[top_5_class_1].values})
table.to_csv("top5_miRNA_active_vs_stable_MS_in_peds.csv", sep=",", index=False)
print("Top 5 features saved to top5_miRNA_active_vs_stable_MS_in_peds.csv.")
