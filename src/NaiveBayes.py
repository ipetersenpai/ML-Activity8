import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("heart_disease.csv")

X = dataset.drop('target', axis=1)
y = dataset['target']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

cm = confusion_matrix(y_test, y_pred, labels=y.unique())
print('Confusion Matrix:')
print(cm)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('Specificity:', specificity)
print('F1 Score:', f1)

# Scatter plot
plt.scatter(X_test['age'], X_test['chol'], c=y_test, cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Scatter Plot of Age vs. Cholesterol with Target Labels')
plt.show()