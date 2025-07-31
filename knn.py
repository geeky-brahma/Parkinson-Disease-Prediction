# import all necessary libraries
import pandas
import sklearn
from sklearn.model_selection import cross_validate,cross_val_score,train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import joblib
import pickle
import os


# load the dataset (local path)
url = "data.csv"
# feature names
features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
dataset = pandas.read_csv(url, names = features)

# store the dataset as an array for easier processing
array = dataset.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(array)
# X stores feature values
X = scaled[:,0:22]
# Y stores "answers", the flower species / class (every row, 4th column)
Y = scaled[:,22]
validation_size = 0.25
# randomize which part of the data is training and which part is validation
seed = 7
# split dataset into training set (80%) and validation set (20%)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size = validation_size, random_state = None)
print(X_train)
# 10-fold cross validation to estimate accuracy (split data into 10 parts; use 9 parts to train and 1 for test)
num_folds = 10
num_instances = len(X_train)
seed = 7
# use the 'accuracy' metric to evaluate models (correct / total)
scoring = 'accuracy'

results = []
clf = KNeighborsClassifier()
# kfold = sklearn.model_selection.KFold(n_splits=num_instances,random_state = seed)
kfold = sklearn.model_selection.KFold(n_splits=num_folds)

cv_results = cross_val_score(clf, X_train, Y_train, cv = kfold, scoring = scoring)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_validation)

# Save the trained model and scaler
print("Saving the KNN model and scaler...")

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Save the model using joblib (recommended for scikit-learn models)
joblib.dump(clf, 'models/knn_model.pkl')

# Save the scaler as well (important for preprocessing new data)
joblib.dump(scaler, 'models/scaler.pkl')

# Alternative: Save using pickle
with open('models/knn_model_pickle.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('models/scaler_pickle.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")
print("📁 Files saved:")
print("  - models/knn_model.pkl (joblib format)")
print("  - models/scaler.pkl (joblib format)")
print("  - models/knn_model_pickle.pkl (pickle format)")
print("  - models/scaler_pickle.pkl (pickle format)")

print("\nKNN Model Performance:")
print("Accuracy:", accuracy_score(Y_validation, predictions)*100, "%")
print("Matthews Correlation Coefficient:", matthews_corrcoef(Y_validation, predictions))
print("\nClassification Report:")
print(classification_report(Y_validation, predictions))
