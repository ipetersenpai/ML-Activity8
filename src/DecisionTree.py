import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

dataset = pd.read_csv("IRIS_dataset.csv")

X = dataset.drop('species', axis=1)
y = dataset['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

cm = confusion_matrix(y_test, y_pred, labels=y.unique())
print('Confusion Matrix:')
print(cm)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # Specificity for the second class (index 1)
f1 = f1_score(y_test, y_pred, average='weighted')

print('\nPrecision:', precision)
print('Recall:', recall)
print('Specificity:', specificity)
print('F1 Score:', f1)

plt.figure(figsize=(10, 8))
tree.plot_tree(dtc, feature_names=X.columns, class_names=y.unique(), filled=True)
plt.show()
