import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data():
    column_names = [f'V{i}' for i in range(1, 87)]
    df = pd.read_csv('data/processed/insurance_cleaned.csv')
    X = df.drop(columns=['V86'])
    y = df['V86']

    return X, y

def build_preprocessor(X):
    numerical_features = list(X.columns)
    preprocessor = ColumnTransformer([
        ('num' , StandardScaler() , numerical_features)
    ])
    return preprocessor



