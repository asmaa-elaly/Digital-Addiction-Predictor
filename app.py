import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# 1. Configuration and Loading Assets
# ----------------------------------------------------

# Set page configuration for a clean look
st.set_page_config(
    page_title="🤳💉 Phone Addiction Predictor & Analysis",
    layout="wide", # Changed to 'wide' for better visualization display
    initial_sidebar_state="collapsed",
)

# Use caching to load the model, scaler, and features only once
@st.cache_resource
def load_assets():
    """Loads the trained model, scaler, and feature list from joblib files.
    Crucially, it now performs Feature Engineering on the main EDA DataFrame (df)
    to make the new features available globally across the app.
    """
    try:
        model = joblib.load('final_xgb_model.joblib')
        scaler = joblib.load('standard_scaler.joblib')
        features = joblib.load('model_features.joblib')
        
        # Load the original data for EDA
        df = pd.read_csv('teen_phone_addiction_dataset.csv')
        df.drop(columns=['ID', 'Name', 'Location', 'Age', 'School_Grade','Parental_Control'], inplace=True)
        
        # --- GLOBAL FEATURE ENGINEERING FOR EDA ---
        # 1. Checks per Hour: Phone_Checks_Per_Day / (Daily_Usage_Hours + 1)
        df['Check_Per_Hour'] = df['Phone_Checks_Per_Day'] / (df['Daily_Usage_Hours'] + 1)
        # 2. App Switching Rate: Apps_Used_Daily / (Daily_Usage_Hours + 1)
        df['App_Switching_Rate'] = df['Apps_Used_Daily'] / (df['Daily_Usage_Hours'] + 1)
        # 3. Usage Sleep Ratio: Daily_Usage_Hours / (Sleep_Hours + 1)
        df['Usage_Sleep_Ratio'] = df['Daily_Usage_Hours'] / (df['Sleep_Hours'] + 1)
        # ------------------------------------------

        return model, scaler, features, df
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Ensure 'teen_phone_addiction_dataset.csv' is present and you have run 'python save_model.py'.")
        return None, None, None, None

model, scaler, feature_names, df = load_assets()

if model is None:
    st.stop() # Stop the application if assets failed to load

# ----------------------------------------------------
# 2. Feature Definitions
# ----------------------------------------------------

NUMERIC_FEATURES = [
    'Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance', 'Social_Interactions',
    'Exercise_Hours', 'Anxiety_Level', 'Depression_Level', 'Self_Esteem',
    'Parental_Control', 'Screen_Time_Before_Bed', 'Phone_Checks_Per_Day',
    'Apps_Used_Daily', 'Time_on_Social_Media', 'Time_on_Gaming',
    'Time_on_Education', 'Family_Communication', 'Weekend_Usage_Hours',
    # --- New Engineered Features added to be scaled correctly ---
    'Check_Per_Hour', 
    'App_Switching_Rate', 
    'Usage_Sleep_Ratio'
]

CATEGORICAL_FEATURES = {
    'Gender': ['Male', 'Female', 'Other'],
    'Phone_Usage_Purpose': ['Browsing', 'Education', 'Gaming', 'Social Media']
}



# ----------------------------------------------------
# 3. Dedicated Functions for Each Tab
# ----------------------------------------------------

def prediction_tab_content():
    """Generates the content for the Prediction Tab (Inputs and Output)."""
    st.header("Predict Phone Addiction Level")
    st.markdown("Enter the teen's profile and behavioral data to predict the Phone Addiction Level (1.0 - 10.0).")

    with st.container(border=True):
        st.subheader("Profile & Usage Data")

        # Use three columns for better visual organization and responsiveness
        col1, col2, col3 = st.columns(3)

        # --- Column 1: Core Usage & Profile ---
        with col1:
            gender = st.selectbox("1. Gender", CATEGORICAL_FEATURES['Gender'], key='gender')
            daily_usage = st.slider("2. Daily Usage (Hours)", 1.0, 10.0, 4.0, 0.1, key='daily_usage')
            sleep_hours = st.slider("3. Sleep (Hours)", 3.0, 10.0, 7.0, 0.1, key='sleep_hours')
            academic_performance = st.slider("4. Academic Performance (0-100)", 0, 100, 75, 1, key='acad_perf')
            family_communication = st.number_input("17. Family Communication (1-10)", min_value=1, max_value=10, value=7, step=1, key='family_comm')


        # --- Column 2: Psychological & Social Data ---
        with col2:
            anxiety_level = st.slider("7. Anxiety Level (1-10)", 1, 10, 5, key='anxiety')
            depression_level = st.slider("8. Depression Level (1-10)", 1, 10, 5, key='depression')
            self_esteem = st.slider("9. Self Esteem (1-10)", 1, 10, 6, key='self_esteem')
            social_interactions = st.slider("5. Social Interactions (1-10)", 1, 10, 5, key='social_int')
            exercise_hours = st.number_input("6. Exercise (Hours/Day)", 0.0, 4.0, 1.0, 0.1, key='exercise')


        # --- Column 3: Specific Phone Behaviors ---
        with col3:
            screen_time_before_bed = st.slider("10. Screen Time Before Bed (Hours)", 0.0, 2.6, 0.5, 0.1, key='screen_bed')
            phone_checks_per_day = st.number_input("11. Phone Checks Per Day", min_value=20, max_value=150, value=80, step=5, key='phone_checks')
            apps_used_daily = st.number_input("12. Apps Used Daily", min_value=5, max_value=20, value=10, step=1, key='apps_daily')
            weekend_usage_hours = st.slider("18. Weekend Usage (Hours)", 0.0, 14.0, 7.0, 0.5, key='weekend_usage')
            usage_purpose = st.selectbox("16. Main Usage Purpose", CATEGORICAL_FEATURES['Phone_Usage_Purpose'], key='usage_purpose')

        st.markdown("---")
        st.subheader("Time Allocation (Hours/Day)")

        colA, colB, colC = st.columns(3)
        with colA:
            time_social_media = st.slider("13. Social Media Time", 0.0, 5.0, 1.5, 0.1, key='time_social')
        with colB:
            time_on_gaming = st.slider("14. Gaming Time", 0.0, 5.0, 1.0, 0.1, key='time_gaming')
        with colC:
            time_on_education = st.slider("15. Education Time", 0.0, 5.0, 0.5, 0.1, key='time_edu')


    # ----------------------------------------------------
    # Prediction Logic
    # ----------------------------------------------------
    if st.button("📈 Predict Addiction Level", use_container_width=True, type="primary"):
        raw_input = {
            'Gender': gender,
            'Daily_Usage_Hours': daily_usage,
            'Sleep_Hours': sleep_hours,
            'Academic_Performance': academic_performance,
            'Social_Interactions': social_interactions,
            'Exercise_Hours': exercise_hours,
            'Anxiety_Level': anxiety_level,
            'Depression_Level': depression_level,
            'Self_Esteem': self_esteem,
            'Screen_Time_Before_Bed': screen_time_before_bed,
            'Phone_Checks_Per_Day': phone_checks_per_day,
            'Apps_Used_Daily': apps_used_daily,
            'Time_on_Social_Media': time_social_media,
            'Time_on_Gaming': time_on_gaming,
            'Time_on_Education': time_on_education,
            'Phone_Usage_Purpose': usage_purpose,
            'Family_Communication': family_communication,
            'Weekend_Usage_Hours': weekend_usage_hours,
        }

        input_df = pd.DataFrame([raw_input])
        
        # --- FEATURE ENGINEERING (Matching Notebook) ---
        input_df['Check_Per_Hour'] = input_df['Phone_Checks_Per_Day'] / (input_df['Daily_Usage_Hours'] + 1)
        input_df['App_Switching_Rate'] = input_df['Apps_Used_Daily'] / (input_df['Daily_Usage_Hours'] + 1)
        input_df['Usage_Sleep_Ratio'] = input_df['Daily_Usage_Hours'] / (input_df['Sleep_Hours'] + 1)

        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES.keys(), drop_first=True)

        final_input = pd.DataFrame(0, index=[0], columns=feature_names)
        for col in input_encoded.columns:
            if col in feature_names:
                final_input[col] = input_encoded[col].iloc[0]

        # Ensure scaling includes the new engineered features
        numeric_cols_for_scaling = [col for col in NUMERIC_FEATURES if col in final_input.columns]
        final_input[numeric_cols_for_scaling] = scaler.transform(final_input[numeric_cols_for_scaling])

        prediction = model.predict(final_input)[0]

        # Display result button
        st.subheader("Prediction Result")
        
        if prediction < 3.0:
            st.success(f"Addiction Level: {prediction:.2f} (Low Risk)")
            st.balloons()
        elif prediction < 6.5:
            st.warning(f"Addiction Level: {prediction:.2f} (Moderate Risk)")
            st.info("The prediction suggests habits that could develop into addiction. Regular monitoring is recommended.")
        else:
            st.error(f"Addiction Level: {prediction:.2f} (High Risk)") 
            st.markdown("**Immediate attention and intervention strategies may be necessary to address potential phone addiction.**")

        st.markdown("---")
        st.markdown(f"**Raw Predicted Score:** `{prediction:.2f}` (Range 1.0 to 10.0)")


def eda_tab_content(data, xgb_model):
    """Generates the content for the Data Exploration Tab (Visualizations)."""
    st.header("Data Exploration & Model Insights")
    st.markdown("Review the underlying data distributions and see which features the trained model considers most important.")

    # --- 1. Target Variable Distribution (Static Plot, kept for simple distribution) ---
    st.subheader("1. Target Variable")
    st.info("Target Variable in the orginial dataset.")
    
    # We use a simple Streamlit chart here for quick display
    st.bar_chart(data['Addiction_Level'].value_counts().sort_index()) 
    st.markdown("---")
    
    # --- 2. Correlation Heatmap (Interactive Plotly) ---
    st.subheader("2. Feature Correlation Heatmap")
    st.info("Visualizing the linear relationships between all numeric features. Hover over a square to see the correlation coefficient. This now includes the new engineered features.")
    
    # Identify numeric columns for the heatmap
    # NOTE: The df now contains the engineered features because they were added in load_assets
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    correlation_matrix = data[numeric_cols].corr()

    # Plotly Heatmap for interactivity
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=".2f", # Show text labels inside cells
        aspect="auto",
        color_continuous_scale='RdBu_r', # Corresponds to coolwarm/RdBu
        x=correlation_matrix.columns.tolist(),
        y=correlation_matrix.index.tolist()
    )
    # Update layout for better viewing
    fig_corr.update_layout(height=800, width=1000)
    fig_corr.update_xaxes(side="top")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # --- 3. Daily Usage Boxplot (Interactive Plotly) ---
    st.subheader("3. Daily Usage vs. Addiction Level")
    st.info("This box plot shows how the distribution of daily phone usage hours changes as the reported addiction level increases. Hover over the boxes to see quartile values.")
    
    # Bin Addiction_Level for a cleaner categorical plot
    data['Addiction_Risk_Group'] = pd.cut(data['Addiction_Level'], bins=[0, 3, 6.5, 10], labels=['Low (1-3)', 'Moderate (3-6.5)', 'High (6.5-10)'], right=True)
    
    # Plotly Box Plot for interactivity
    fig_box = px.box(
        data, 
        x='Addiction_Risk_Group', 
        y='Daily_Usage_Hours', 
        color='Addiction_Risk_Group', 
        title='Daily Usage Hours Distribution by Addiction Risk Group',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_box.update_layout(xaxis_title='Addiction Risk Group', yaxis_title='Daily Usage Hours')
    st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # --- 4. Top Feature Importances (Existing Plot) ---
    st.subheader("4. Top Feature Importances (from XGBoost Model)")
    st.info("Features with higher importance scores had a greater impact on the model's predictions.")
    
    if hasattr(xgb_model, 'feature_importances_'):
        importances = pd.Series(xgb_model.feature_importances_, index=feature_names)
        
        grouped_importances = {}
        for feature, imp in importances.items():
            # Handling original and one-hot encoded features
            base_feature = feature.split('_')[0] if feature.startswith(tuple(CATEGORICAL_FEATURES.keys())) else feature
            grouped_importances[base_feature] = grouped_importances.get(base_feature, 0) + imp
            
        top_features = pd.Series(grouped_importances).sort_values(ascending=False).head(10)
        
        # Plotly Bar Chart for interactivity (replaces the Matplotlib plot)
        fig2 = px.bar(
            x=top_features.values, 
            y=top_features.index, 
            orientation='h', 
            title='Top 10 Important Features for Addiction Prediction',
            labels={'x': 'Feature Importance Score', 'y': 'Feature'},
            color=top_features.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        fig2.update_yaxes(autorange="reversed") # To display the highest importance at the top
        fig2.update_traces(hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Feature importance data is not available from the loaded model.")
        
    st.markdown("---")
    
    # --- 5. Model Performance Metrics Comparison (Interactive Plotly) ---
    st.subheader("5. Model Performance Comparison")
    st.warning("NOTE: The metrics for Linear Regression and Random Forest are placeholder values. Replace them with your actual results from your training notebook.")
    
    # Placeholder values for demonstration (REPLACE THESE WITH YOUR ACTUAL NOTEBOOK VALUES)
    mae_lin =0.6241795816100523
    rmse_lin = 0.782650238279498
    r2_lin = 0.7568617549910496
    
    mae_rf = 0.3449900000000001
    rmse_rf = 0.5259850473159859
    r2_rf = 0.8901842503902849
    
    # Assuming you retrieved your final XGBoost metrics
    mae_xgb_final = 0.23531247107187905
    rmse_xgb_final = 0.33530684339987765
    r2_xgb_final = 0.955372488724678
    
    # Re-structure data for Plotly grouped bar
    models = ['Linear Regression', 'Random Forest', 'XGBoost']
    metrics_labels = ["MAE", "RMSE", "R2"]
    
    metrics_data = {
        'Model': np.repeat(models, len(metrics_labels)),
        'Metric': metrics_labels * len(models),
        'Value': [mae_lin, rmse_lin, r2_lin, mae_rf, rmse_rf, r2_rf, mae_xgb_final, rmse_xgb_final, r2_xgb_final]
    }
    metrics_df = pd.DataFrame(metrics_data)

    # Plotly Bar Chart for interactivity
    fig_metrics = px.bar(
        metrics_df, 
        x='Metric', 
        y='Value', 
        color='Model', 
        barmode='group',
        text='Value', # This adds text labels to the bars
        title='Model Performance Metrics Comparison'
    )
    # Customize hover template to show precise value on hover
    fig_metrics.update_traces(texttemplate='%{y:.3f}', textposition='outside', hovertemplate='%{y:.4f}<extra></extra>')
    fig_metrics.update_layout(yaxis_title="Metric Value", xaxis_title="Metric")
    
    st.plotly_chart(fig_metrics, use_container_width=True)
    
# ----------------------------------------------------
# 4. Main App Structure (Tabs)
# ----------------------------------------------------

st.title("Phone Addiction Model Dashboard")

tab1, tab2 = st.tabs(["📊 Prediction Tool", "🔍 Data Exploration"])

with tab1:
    prediction_tab_content()

with tab2:
    eda_tab_content(df, model)