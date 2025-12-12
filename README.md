# Digital-Addiction-Predictor

**Project Overview:**
This interactive application, built on Streamlit, is designed to help identify and assess the risk of mobile phone addiction in teenagers. It utilizes an advanced Machine Learning model (XGBoost Regressor) trained on a comprehensive dataset that includes daily usage metrics, psychological factors (anxiety, depression, self-esteem), and educational/social behaviors.

**Features:**
-Data Inspection & Preprocessing
-Feature Engineering such as (Check_Per_Hour and Usage_Sleep_Ratio)
-Exploratory Data Analysis (EDA): Provides a dedicated tab for data exploration, including:

   Interactive Correlation Matrix (Plotly): To visualize the relationships between all numeric features.

   Feature Importance: To identify the features that the XGBoost model relied on most heavily in making its predictions.

   Model Performance Comparison: A comparative display of performance metrics between different models (such as Regression, Random Forest, and XGBoost).ML Modeling (XGBoost, Random Forest, Linear Regression)
-ML Modeling (XGBoost, Random Forest, Linear Regression)

**Technologies Used**
-Interface: Streamlit

-Data Visualization: Plotly (Interactive Charts)

-Machine Learning Model: XGBoost Regressor

-Data Management: Pandas, Joblib
