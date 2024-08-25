import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df):
    # Handle missing values
    df = df.dropna()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['price'], inplace=True)
    df = df[['price', 'area', 'bedrooms', 'bathrooms']]
    X = df[['area', 'bedrooms', 'bathrooms']]
    y = df['price']

    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    df = load_data('../data/housing.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Optional: Save the preprocessed data for later use
    X_train.to_csv('../data/X_train.csv', index=False)
    X_test.to_csv('../data/X_test.csv', index=False)
    y_train.to_csv('../data/y_train.csv', index=False)
    y_test.to_csv('../data/y_test.csv', index=False)