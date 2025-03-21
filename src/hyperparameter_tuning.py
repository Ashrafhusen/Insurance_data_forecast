import optuna
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from src.data_preprocessing import load_data, build_preprocessor


X, y = load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

preprocessor = build_preprocessor(X_train)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
    }
    
    model = Pipeline([
        ('preprocessing' , preprocessor),
        ('xgb' , XGBRegressor(objectives = 'reg:squarederror' , random_state=42 , **params))
    ])


    model.fit(X_train , y_train)
    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse


study = optuna.create_study(direction = 'minimize')
study.optimize(objective , n_trials=30)

print('HyperParameters : ')
for key, val in study.best_params.items():
    print(f"{key} : {val}")


best_params = study.best_params
best_model = Pipeline([
    ('preprocessing' , preprocessor),
    ('xgb', XGBRegressor(objective = 'reg:squarederror', random_state=42, **best_params))
])

best_model.fit(X_train, y_train)
joblib.dump(best_model, 'models/insurance_pricing_model.pkl')
print("model saved to models/insurance_pricing_model.pkl")

import optuna.visualization as vis
vis.plot_optimization_history(study).show()

fig = vis.plot_optimization_history(study)
fig.write_html("reports/optuna_optimization_history.html")
print("Optimization history saved to reports/optuna_optimization_history.html")
