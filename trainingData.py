import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Loading and reading the dataset
heart_df = pd.read_csv("heart.csv")

print(heart_df.describe())
variables = heart_df.columns[:-1]
# Vẽ pairplot cho tất cả các biến
# sns.pairplot(heart_df)
# plt.show()

# Applying IQR method for outlier detection and noise reduction
Q1 = heart_df.quantile(0.25)
Q3 = heart_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("Lower Bound:")
print(lower_bound)
print("Upper Bound:")
print(upper_bound)

print((heart_df < lower_bound).any(axis=1))
print((heart_df > upper_bound).any(axis=1))
outliers = ((heart_df < lower_bound) | (heart_df > upper_bound)).any(axis=1)
outliers_df = heart_df[outliers]
inliers_df = heart_df[~outliers]
print("Outliers:")
print(outliers_df)

heart_df = heart_df[((heart_df >= lower_bound) & (heart_df <= upper_bound)).all(axis=1)]

# Splitting the data into features (X) and target (y)
X = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the Random Forest classifier
model = RandomForestClassifier(n_estimators=30)
model.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating the model
classification_rep = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print('Classification Report:\n', classification_rep)
print('Accuracy: {}%\n'.format(round(accuracy * 100, 2)))
print('Confusion Matrix:\n', confusion_mat)

# Saving the model to a file using pickle
filename = 'heart-disease-prediction-model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Extracting feature importances from the trained model
importances = model.feature_importances_

# Sorting feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearranging feature names based on feature importances
feature_names = X.columns[indices]

# Creating a bar plot of feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names, rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()