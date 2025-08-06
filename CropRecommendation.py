import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# Features (independent variables) and Target (dependent variable)
x = df.drop(['label'], axis=1)   # All columns except 'label'
y = df['label']                  # Crop label (target variable)

# Encode crop labels into numeric values for ML models
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Crop to numeric mapping:", label_mapping)

# Standardize feature values for better model performance
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
x_scaled = scalar.fit_transform(x)

# Split data into training and testing sets (67% train, 33% test)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y_encoded, test_size=1/3, random_state=42
)

# ---------------------------
# 1. Random Forest Classifier
# ---------------------------
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, criterion='entropy')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", score)

# ---------------------------
# 2. Support Vector Machine (SVM)
# ---------------------------
from sklearn.svm import SVC
clf = SVC(kernel='rbf', random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("SVM Confusion Matrix:\n", cm)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 3. Multi-Layer Perceptron (Neural Network)
# ---------------------------
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # 2 hidden layers: 128 & 64 neurons
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)
print("MLP Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 4. Polynomial Regression (Not ideal for classification)
# ---------------------------
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

poly_reg = PolynomialFeatures(degree=4)
x_poly_train = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly_train, y_train)

x_poly_test = poly_reg.transform(x_test)
y_pred = lin_reg2.predict(x_poly_test)
print("Polynomial Regression R2 Score:", r2_score(y_test, y_pred))

# ---------------------------
# 5. K-Nearest Neighbors (KNN)
# ---------------------------
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("KNN Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 6. Naive Bayes Classifier
# ---------------------------
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# 7. Decision Tree (Regressor + Classifier)
# ---------------------------
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Decision Tree as Regressor (for R2 score)
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print("Decision Tree Regressor R2 Score:", r2_score(y_test, y_pred))

# Decision Tree Classifier
clf = DecisionTreeClassifier()
y_pred = clf.fit(x_train, y_train).predict(x_test)
print("Decision Tree Classifier Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Decision Tree Classifier Accuracy:", accuracy_score(y_test, y_pred))

# ---------------------------
# Test user input using trained MLP model
# ---------------------------
lst = []
lstn = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']

# Take 7 inputs for the features
for feature in lstn:
    lst.append(float(input(f'Enter {feature}: ')))

# Scale user input and predict recommended crop
lst_scaled = scalar.transform([lst])
test_sample = np.array(lst_scaled)
prediction = mlp.predict(test_sample)
predicted_crop = le.inverse_transform(prediction)
print("Recommended crop:", predicted_crop[0])
