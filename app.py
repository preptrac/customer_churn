import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error loading the dataset: {e}")
            return None
    else:
        st.warning("Please upload a dataset to proceed.")
        return None

# Preprocessing
@st.cache_data  # Cache the preprocessing step
def preprocess_data(data):
    if data is None:
        return None

    # Convert 'TotalCharges' to numeric, handling errors
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    # Drop non-numeric columns like 'customerID' (if they exist)
    if 'customerID' in data.columns:
        data.drop(columns=['customerID'], inplace=True)
    
    # Convert categorical variables to dummy variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                           'MultipleLines', 'InternetService', 'OnlineSecurity', 
                           'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                           'StreamingTV', 'StreamingMovies', 'Contract', 
                           'PaperlessBilling', 'PaymentMethod']
    
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    # Convert target variable 'Churn' to binary
    data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    return data

# Streamlit App
st.title("Telco Customer Churn Analysis")

# File uploader outside the cached function
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Load and preprocess data
if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is not None:
        data = preprocess_data(data)

        # Display the dataset
        st.write("### Dataset Preview")
        st.write(data.head())

        # Visualizations
        st.write("### Customer Demographics and Service Usage Patterns")

        # Select columns for visualization
        columns = data.columns.tolist()
        selected_columns = st.multiselect("Select columns to visualize", columns)

        # Plot selected columns
        if selected_columns:
            for col in selected_columns:
                st.write(f"#### Distribution of {col}")
                fig, ax = plt.subplots()
                if data[col].dtype in ['int64', 'float64']:
                    sns.histplot(data[col], kde=True, ax=ax)
                else:
                    sns.countplot(data[col], ax=ax)
                st.pyplot(fig)

        # Split data into features and target
        X = data.drop('Churn', axis=1)
        y = data['Churn']

        # Ensure all feature columns are numeric
        X = X.astype(float)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest Classifier
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### Model Accuracy: {accuracy:.2f}")
        st.write("### Classification Report")
        st.write(classification_report(y_test, y_pred))

        # Real-time prediction
        st.write("### Real-Time Churn Prediction")
        st.write("Select a customer to predict churn:")

        # Add a dropdown to select a customer
        customer_ids = data.index.tolist()
        selected_customer = st.selectbox("Select Customer ID", customer_ids)
        
        # Display selected customer's data
        st.write("Selected Customer Data:")
        st.write(data.loc[selected_customer])

        # Predict churn for the selected customer
        if st.button("Predict Churn"):
            input_data = data.loc[selected_customer].drop('Churn')
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write(f"### Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")

        # Interactive data filtering
        st.write("### Interactive Data Filtering")
        filter_column = st.selectbox("Select column to filter by", columns)

        if data[filter_column].dtype in ['int64', 'float64']:
            min_val = data[filter_column].min()
            max_val = data[filter_column].max()
            filter_value = st.slider(f"Select range for {filter_column}", min_val, max_val, (min_val, max_val))
            filtered_data = data[(data[filter_column] >= filter_value[0]) & (data[filter_column] <= filter_value[1])]
        else:
            filter_value = st.text_input(f"Enter value for {filter_column}")
            filtered_data = data[data[filter_column].astype(str).str.contains(filter_value, case=False)]

        st.write(filtered_data)