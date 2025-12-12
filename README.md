# Digital Addiction Predictor

---

This interactive application, built on Streamlit, is designed to help identify and assess the risk of mobile phone addiction in teenagers. It utilizes an advanced Machine Learning model (XGBoost Regressor) trained on a comprehensive dataset that includes daily usage metrics, psychological factors (anxiety, depression, self-esteem), and educational/social behaviors.

---

##  Features
- Prediction Tool: Allows users to input specific teenager data (such as sleep hours, academic performance, and phone checking rate) to obtain an accurate predictive score for the addiction level (ranging from 1.0 to 10.0).

- Feature Engineering: The model employs newly engineered features (such as Check_Per_Hour and Usage_Sleep_Ratio) to enhance prediction accuracy, reflecting a deeper understanding of digital behavior patterns.

- Exploratory Data Analysis (EDA): Provides a dedicated tab for data exploration, including:

- Interactive Correlation Matrix (Plotly): To visualize the relationships between all numeric features.

- Feature Importance: To identify the features that the XGBoost model relied on most heavily in making its predictions.

- Model Performance Comparison: A comparative display of performance metrics between different models (such as Regression, Random Forest, and XGBoost).

---

##  Project Structure
```
phone_addiction_predictor/
│
├── app.py                      # Main Streamlit Dashboard, Prediction Logic, and EDA
├── requirements.txt            # Python dependencies required for the project
├── README_description_en.md    # Comprehensive project overview and documentation
├── save_model.py               3saving model metrices
├── teen_phone_addiction_dataset.csv # The raw dataset used for training and EDA
├── final_xgb_model.joblib      # The trained XGBoost Regressor model artifact
├── standard_scaler.joblib      # The pre-fitted Standard Scaler artifact
└── model_features.joblib       # List of expected feature names (for input integrity)
```

---


## Model Performance
| Model              | R² Score | MAE        |
|-------------------|----------|------------ |
| Linear Regression | 0.757    | .624        |
| Random Forest     | 0.890    | .345     |
| **XGBoost**       | **.995** | **.235** |

---

##  Future Improvements
- Deploy a mobile application to detect the inputs automatically

---

##  Visual Examples 
- Correlation heatmap
- Feature importance plot
- Model comparison bar chart

---

##  Technologies Used

- **Python 3.10** – Core programming language
- **Pandas / NumPy** – Data manipulation
- **Matplotlib / Seaborn / Plotly** – Visualization
- **Scikit-learn** – Machine learning algorithms & preprocessing
- **XGBoost / Random Forest / Linear Regression** – Modeling
- **Streamlit** – Interactive web application
- **Git + GitHub** – Version control & collaboration
