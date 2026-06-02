# IMPORT LIBRARIES
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. LOAD DATASET
iris = load_iris()
X = iris.data          # Features (sepal/petal measurements)
y = iris.target        # Labels (3 flower species)

# 2. SPLIT DATA (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. SCALE FEATURES (CRITICAL for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Learn scaling from train data
X_test = scaler.transform(X_test)         # Apply same scaling to test data

# 4. TRAIN KNN MODEL 
for k in [1, 3, 5, 7, 9]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    print(f"K={k} → Accuracy: {model.score(X_test, y_test):.2%}")
# 5. PREDICT
predictions = model.predict(X_test)

# 6. EVALUATE
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report (Precision, Recall, F1-score):")
print(classification_report(y_test, predictions, target_names=iris.target_names))
print(f"Accuracy: {model.score(X_test, y_test):.2%}")