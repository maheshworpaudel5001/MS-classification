import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 0) prepare the data
df = pd.read_excel(
    "/home/gddaslab/mxp140/sclerosis_project/miRNA_signal_hsa_number2.xlsx",
    engine="openpyxl",
    sheet_name="Sheet1",
)

# select all rows and only the columns whose names contain either 'sPOMS' or 'aPOMS'
data = df.loc[:, df.columns.str.contains('sPOMS') | df.columns.str.contains('aPOMS')]

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

# 1) model
model = RandomForestClassifier(n_estimators=100, random_state=1234)

# 2) Training and evaluation using cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# Train the model on the entire training set
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy:.2f}")

# Feature importance
feature_importances = model.feature_importances_

# Get the indices of the top 5 features for class 1 (highest positive importances)
top_5_class_1 = np.argsort(feature_importances)[-5:]

# Get the indices of the top 5 features for class 0 (highest negative importances)
top_5_class_0 = np.argsort(feature_importances)[:5]

print("Top 5 features contributing to class 1:", top_5_class_1)
print("Top 5 features contributing to class 0:", top_5_class_0)

# Optionally, you can print the feature names if you have them
feature_names = df['Transcript_ID']
print("Top 5 features contributing to class 1:", feature_names[top_5_class_1].values)
print("Top 5 features contributing to class 0:", feature_names[top_5_class_0].values)

table = pd.DataFrame({"sPOMS": feature_names[top_5_class_0].values, "aPOMS": feature_names[top_5_class_1].values})
table.to_csv("top5_important_features_random_forest.csv", sep=",", index=False)
print("Top 5 features saved to top5_important_features_random_forest.csv.")