import joblib
import shap
from src.data_preprocessing import load_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

model = joblib.load('models/insurance_pricing_model.pkl')
X, y = load_data()
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state = 42)


explainer = shap.Explainer(model.named_steps['xgb'])
shap_values = explainer(X_test)

shap.summary_plot(shap_values , X_test)
