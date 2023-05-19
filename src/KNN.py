import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("heart_disease.csv")

X = dataset.drop('target', axis=1)
y = dataset['target']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Create a K-Nearest Neighbors classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
f1 = f1_score(y_test, y_pred, average='macro')

print('Precision:', precision)
print('Recall:', recall)
print('Specificity:', specificity)
print('F1 Score:', f1)

# Scatter plot
plt.scatter(X_test['age'], X_test['chol'], c=y_test, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Scatter Plot of Age vs. Cholesterol with KNN Predictions')
plt.show()
