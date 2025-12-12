import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor 
import numpy as np 
BEST_PARAMS = {
    'colsample_bytree': 0.7923575302488596,
    'learning_rate': 0.04656407456416188,
    'max_depth': 5,
    'n_estimators': 393,
    'subsample': 0.6
}
RANDOM_SEED = 42

try:
    df = pd.read_csv('teen_phone_addiction_dataset.csv') 
except FileNotFoundError:
    print("Error: 'teen_phone_addiction_dataset.csv' not found. Please place the data file in the project folder.")
    exit()

df.drop(columns=['ID', 'Name', 'Location', 'Age', 'School_Grade','Parental_Control'], inplace=True)
X = df.drop('Addiction_Level', axis=1)
y = df['Addiction_Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

numeric_cols = X_train.select_dtypes(include=np.number).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for c in missing_cols:
    X_test_encoded[c] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]



scaler = StandardScaler()
X_train_encoded[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])
X_test_encoded[numeric_cols] = scaler.transform(X_test_encoded[numeric_cols])

joblib.dump(scaler, 'standard_scaler.joblib')
print("StandardScaler saved as standard_scaler.joblib")

final_model = XGBRegressor(**BEST_PARAMS, random_state=RANDOM_SEED)

final_model.fit(X_train_encoded, y_train)

joblib.dump(final_model, 'final_xgb_model.joblib')
print("XGBoost model saved as final_xgb_model.joblib")

joblib.dump(X_train_encoded.columns.tolist(), 'model_features.joblib')
print("Feature column names saved as model_features.joblib")