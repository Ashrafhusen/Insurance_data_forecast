import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb
from src.data_preprocessing import load_data, build_preprocessor


X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2 , random_state=42)

preprocessor = build_preprocessor(X)

model = Pipeline([
    ('preprocessing' , preprocessor),
    ('xgb' , xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ))
])



model.fit(X_train, y_train)
joblib.dump(model, 'models/insurance_pricing_model.pkl')
print('Model Trained and Saved!')

