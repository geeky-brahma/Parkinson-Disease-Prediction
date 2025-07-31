"""
Model Prediction Script
======================
This script demonstrates how to load the saved KNN model and make predictions
on new data for Parkinson's Disease diagnosis.
"""

import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

def load_model_joblib():
    """Load the KNN model and scaler using joblib"""
    try:
        model = joblib.load('models/knn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model and scaler loaded successfully using joblib!")
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Please run knn.py first to train and save the model.")
        return None, None

def load_model_pickle():
    """Load the KNN model and scaler using pickle"""
    try:
        with open('models/knn_model_pickle.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler_pickle.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Model and scaler loaded successfully using pickle!")
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Please run knn.py first to train and save the model.")
        return None, None

def predict_parkinsons(model, scaler, features):
    """
    Predict Parkinson's disease for given features
    
    Parameters:
    model: Trained KNN model
    scaler: Fitted MinMaxScaler
    features: List or array of 22 voice features
    
    Returns:
    prediction: 0 (No Parkinson's) or 1 (Parkinson's)
    probability: Confidence score
    """
    # Reshape features for single prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Scale the features using the same scaler used during training
    scaled_features = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(scaled_features[:, :22])  # Use only first 22 features
    
    # Get prediction probabilities (confidence)
    try:
        probabilities = model.predict_proba(scaled_features[:, :22])
        confidence = max(probabilities[0])
    except AttributeError:
        # If predict_proba is not available, use distance-based confidence
        distances, indices = model.kneighbors(scaled_features[:, :22])
        confidence = 1 / (1 + np.mean(distances))
    
    return prediction[0], confidence

def test_model_on_dataset():
    """Test the loaded model on the original dataset"""
    # Load the model
    model, scaler = load_model_joblib()
    if model is None:
        return
    
    # Load the test data
    try:
        url = "data.csv"
        features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE","status"]
        dataset = pd.read_csv(url, names=features)
        
        # Prepare data (same preprocessing as in training)
        array = dataset.values
        scaled = scaler.transform(array)  # Use transform, not fit_transform
        X_test = scaled[:, :22]
        Y_test = scaled[:, 22]
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate performance
        accuracy = accuracy_score(Y_test, predictions) * 100
        mcc = matthews_corrcoef(Y_test, predictions)
        
        print(f"\n🔬 Model Test Results:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        
    except FileNotFoundError:
        print("❌ data.csv not found. Cannot test model.")

def example_prediction():
    """Example of making a prediction with sample data"""
    model, scaler = load_model_joblib()
    if model is None:
        return
    
    # Example features (you would replace these with actual voice measurements)
    sample_features = [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
        0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
        0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654, 1
    ]
    
    prediction, confidence = predict_parkinsons(model, scaler, sample_features)
    
    print(f"\n🎯 Example Prediction:")
    print(f"Input features: {sample_features[:5]}... (showing first 5 of 23)")
    
    if prediction == 1:
        result = "Parkinson's Disease"
    else:
        result = "No Parkinson's Disease"
    
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    print("🧠 Parkinson's Disease Prediction - Model Loader")
    print("=" * 50)
    
    # Test loading with joblib
    print("\n1. Loading model with joblib:")
    load_model_joblib()
    
    # Test loading with pickle
    print("\n2. Loading model with pickle:")
    load_model_pickle()
    
    # Test model performance
    print("\n3. Testing model on dataset:")
    test_model_on_dataset()
    
    # Example prediction
    print("\n4. Example prediction:")
    example_prediction()
    
    print("\n" + "=" * 50)
    print("✨ All tests completed!")
