import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib

from src.data_preprocessing import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


model = joblib.load('models/insurance_pricing_model.pkl')
X, y = load_data()
_, X_test, _, y_test = train_test_split(X , y, test_size=0.2 , random_state=42)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f'MAE : {mae:.2f} | RMSE: {rmse: .2f} | R2 : {r2: .4f}')

