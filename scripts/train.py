import joblib
from sklearn.linear_model import LinearRegression
from preprocess import load_data, preprocess_data
import numpy as np
import os

def train_model(X_train, y_train):
    print("Training the model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

if __name__ == "__main__":
    print("Starting the training process...")

    try:
        df = load_data('C:/Users/ajayn/Desktop/ML/housing_price_prediction/data/Housing.csv')
        print("Data loaded successfully.")
        
        X_train, X_test, y_train, y_test = preprocess_data(df)
        print("Data preprocessing complete.")
        
        model = train_model(X_train, y_train)
        print("Model trained successfully.")
        
       
        
        # Save the model
        joblib.dump(model, os.path.join("C:/Users/ajayn/Desktop/ML/housing_price_prediction/models", 'model.pkl'))
        print("Model saved successfully.")

        score = model.score(X_test, y_test)
        print(f"Model R^2 Score: {score}")

    except Exception as e:
        print(f"An error occurred: {e}")

