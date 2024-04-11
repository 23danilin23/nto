from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
import pandas as pd
import numpy as np
import random

df = pd.read_csv("output_data.csv")

X = [(df["Radius"][i], df["Square"][i], df["Circumference"][i]) for i in range(len(df["Radius"]))]
y = tuple(df["Octane number"].tolist())

# print(len(X), len(y))

seed = 69420

np.random.seed(seed)

random.seed(seed)
'''
#Quadratic discriminant analysis

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Quadratic Discriminant Analysis model
qda = QuadraticDiscriminantAnalysis()

# Train the model
qda.fit(X_train, y_train)

# Make predictions on the test set
y_pred = qda.predict(X_test)

# Calculate evaluation metrics
qda_precision = precision_score(y_test, y_pred, average="weighted")
qda_recall = recall_score(y_test, y_pred, average="weighted")
qda_f1 = f1_score(y_test, y_pred, average="weighted")

print(qda_f1, qda_recall, qda_precision)

#Logistic regression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Logistic Regression model
logistic_reg = LogisticRegression(random_state=seed)

# Train the model
logistic_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_reg.predict(X_test)

# Calculate evaluation metrics
logreg_precision = precision_score(y_test, y_pred, average = "weighted")
logreg_recall = recall_score(y_test, y_pred, average = "weighted")
logreg_f1 = f1_score(y_test, y_pred, average = "weighted")

print("Logistic regression done")

#Decision tree

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=seed)

# Train the model
decision_tree.fit(X_train, y_train)

# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

# Calculate evaluation metrics
dectree_precision = precision_score(y_test, y_pred, average = "weighted")
dectree_recall = recall_score(y_test, y_pred, average = "weighted")
dectree_f1 = f1_score(y_test, y_pred, average = "weighted")

print("dicision tree done")

#Random Forest

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier(random_state=seed)

# Train the model
random_forest.fit(X_train, y_train)

# Make predictions on the test set
y_pred = random_forest.predict(X_test)

# Calculate evaluation metrics
ranfor_precision = precision_score(y_test, y_pred, average = "weighted")
ranfor_recall = recall_score(y_test, y_pred, average = "weighted")
ranfor_f1 = f1_score(y_test, y_pred, average = "weighted")

print("random forest done")

#Support Vector Machines

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Support Vector Machine classifier
svm = SVC(random_state=seed)

# Train the model
svm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test)

# Calculate evaluation metrics
svm_precision = precision_score(y_test, y_pred, average = "weighted")
svm_recall = recall_score(y_test, y_pred, average = "weighted")
svm_f1 = f1_score(y_test, y_pred, average = "weighted")

print("svm done")

#K-Nearest Neighbors

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the K-Nearest Neighbors classifier
knn = KNeighborsClassifier()

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate evaluation metrics
knn_precision = precision_score(y_test, y_pred, average = "weighted")
knn_recall = recall_score(y_test, y_pred, average = "weighted")
knn_f1 = f1_score(y_test, y_pred, average = "weighted")

print("knn done")

#Naive Bayes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Naive Bayes classifier
naive_bayes = GaussianNB()

# Train the model
naive_bayes.fit(X_train, y_train)

# Make predictions on the test set  
y_pred = naive_bayes.predict(X_test)

# Calculate evaluation metrics
nb_precision = precision_score(y_test, y_pred, average = "weighted")
nb_recall = recall_score(y_test, y_pred, average = "weighted")
nb_f1 = f1_score(y_test, y_pred, average = "weighted")

print("nb done")

#Principal Component Analysis

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize PCA
pca = PCA(n_components=2)  # Assuming you want to reduce the features to 2 dimensions tbh idk

# Fit PCA on the training data and transform both training and testing data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize the classification algorithm (SVM in this example)
svm = SVC(random_state=seed)

# Train the model on the PCA-transformed data
svm.fit(X_train_pca, y_train)

# Make predictions on the test set
y_pred = svm.predict(X_test_pca)

# Calculate evaluation metrics
pca_precision = precision_score(y_test, y_pred, average = "weighted")
pca_recall = recall_score(y_test, y_pred, average = "weighted")
pca_f1 = f1_score(y_test, y_pred, average = "weighted")

print("pca done")

#Gradient Boosting Machines

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Initialize the Gradient Boosting Classifier
gbm = GradientBoostingClassifier(random_state=seed)

# Train the model
gbm.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gbm.predict(X_test)

# Calculate evaluation metrics
gbm_precision = precision_score(y_test, y_pred, average = "weighted")
gbm_recall = recall_score(y_test, y_pred, average = "weighted")
gbm_f1 = f1_score(y_test, y_pred, average = "weighted")

print("gbm done")
'''

#Artificial Neural Networks 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

np.random.seed(seed)

tf.random.set_seed(seed)

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, input_dim=np.array(X_train).shape[1], activation='relu'))  # Input layer with 64 neurons and ReLU activation
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons and ReLU activation
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron and sigmoid activation for binary classification

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(type(X_train))
print(type(y_train))

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32)

# Make predictions on the test set
y_pred_prob = np.array(model.predict(i) for i in X_test)
y_pred = np.array(np.where(i > 0.5, 1, 0) for i in y_pred_prob)  # Convert probabilities to binary predictions
   
# Calculate evaluation metrics
ann_precision = precision_score(y_test, y_pred, average = "weighted")
ann_recall = recall_score(y_test, y_pred, average = "weighted")
ann_f1 = f1_score(y_test, y_pred, average = "weighted")

print("ann done")


df = pd.DataFrame({
    "model name" : ("Quadratic discriminant analysis", "Logistic regression", "Decision tree", "Random Forest", "Support Vector Machines", "K-Nearest Neighbors", "Naive Bayes", "Principal Component Analysis", "Gradient Boosting Machines", "Artificial Neural Networks "),
    "precision" : (qda_precision, logreg_precision, dectree_precision, ranfor_precision, svm_precision, knn_precision, nb_precision, pca_precision, gbm_precision, ann_precision),
    "recall" : (qda_recall, logreg_recall, dectree_recall, ranfor_recall, svm_recall, knn_recall, nb_recall, pca_recall, gbm_recall, ann_recall),
    "f1-score" : (qda_f1, logreg_f1, dectree_f1, ranfor_f1, svm_f1, knn_f1, nb_f1, pca_f1, gbm_f1, ann_f1)
})

df.to_csv("output_metrics_nn.csv", index = False)