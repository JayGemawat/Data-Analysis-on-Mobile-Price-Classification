import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Define paths to your local files
train_file_path =r"C:\Users\Diya Patel\Desktop\Data-Analysis-on-Mobile-Price-Classification\train.csv" # Adjust this path if necessary
test_file_path = r"C:\Users\Diya Patel\Desktop\Data-Analysis-on-Mobile-Price-Classification\test.csv"    # Adjust this path if necessary
model_file_path =r"C:\Users\Diya Patel\Desktop\Data-Analysis-on-Mobile-Price-Classification\model.pkl"  # Adjust this path if necessary

# Streamlit App Title
st.title("Mobile Price Classification App")

# Project Description
st.markdown("""
## Project Overview

This research explores predicting mobile phone prices using machine learning techniques. A **Decision Tree Classifier** is applied to classify phones into four price categories based on their features. The dataset used includes specifications such as battery capacity, RAM, and screen resolution. The classifier achieved an accuracy of **80.75%**. This study illustrates how tree-based algorithms can be effective in mobile price prediction, with potential for further improvement through feature refinement and hyperparameter tuning.

### Price Range Categories
The target variable has four distinct values:
- **0**: Represents the **lowest price range** (budget or low-cost phones).
- **1**: Represents the **second price range** (lower mid-range phones).
- **2**: Represents the **third price range** (upper mid-range phones).
- **3**: Represents the **highest price range** (premium or high-cost phones).

These categories indicate which price range a mobile phone falls into based on its technical specifications.
""")

# Load the training and testing datasets
try:
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    train_data = pd.DataFrame()  # Set to empty DataFrame if loading fails
    test_data = pd.DataFrame()    # Set to empty DataFrame if loading fails

# Display training data
if not train_data.empty:
    st.subheader("Training Data")
    st.write(train_data.head())
else:
    st.warning("Training dataset is empty or not loaded.")

# Display testing data
if not test_data.empty:
    st.subheader("Testing Data")
    st.write(test_data.head())
else:
    st.warning("Testing dataset is empty or not loaded.")

# Load the trained model
model = None
try:
    model = joblib.load(model_file_path)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")

# Sidebar - Prediction Inputs
st.sidebar.subheader("Input Features for Prediction")
if not train_data.empty:
    # Automatically get feature columns excluding target ('price_range')
    feature_columns = [col for col in train_data.columns if col != 'price_range']
    input_values = {}

    # Create input fields for each feature in the sidebar
    for feature in feature_columns:
        if pd.api.types.is_numeric_dtype(train_data[feature]):
            input_values[feature] = st.sidebar.number_input(
                f"{feature}",
                min_value=float(train_data[feature].min()),
                max_value=float(train_data[feature].max())
            )
        else:
            input_values[feature] = st.sidebar.selectbox(
                f"{feature}",
                options=train_data[feature].unique()
            )

    # Prediction
    if st.sidebar.button("Predict", key="predict_button"):
        # Prepare the features in the required format for prediction
        features = pd.DataFrame([list(input_values.values())], columns=feature_columns)  # Convert input values to DataFrame
        try:
            prediction = model.predict(features)
            st.sidebar.write(f"Predicted Price Range: {prediction[0]}")
            st.sidebar.markdown("""
            ### Prediction Meaning
            The predicted price range corresponds to:
            - **0**: Budget (low-cost phones)
            - **1**: Lower Mid-Range
            - **2**: Upper Mid-Range
            - **3**: Premium (high-cost phones)
            """)
        except Exception as e:
            st.sidebar.error(f"Error during prediction: {e}")

# Data visualizations
st.subheader("Data Visualizations")

# Price Range Distribution
if not train_data.empty and st.checkbox("Show Price Range Distribution"):
    plt.figure(figsize=(10, 5))
    sns.countplot(data=train_data, x='price_range')
    st.pyplot(plt)
    plt.close()

# Correlation Heatmap
if not train_data.empty and st.checkbox("Show Correlation Heatmap"):
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)
    plt.close()

# Pie Chart - Proportion of Phones in Each Price Range
if not train_data.empty and st.checkbox("Show Proportion of Phones by Price Range"):
    price_range_counts = train_data['price_range'].value_counts()
    fig_pie = px.pie(values=price_range_counts, names=price_range_counts.index, title='Proportion of Phones by Price Range')
    st.plotly_chart(fig_pie)

# Pairwise Correlation with Target Variable
if not train_data.empty and st.checkbox("Show Correlation with Price Range"):
    st.subheader("Feature Correlation with Price Range")
    corr_with_target = train_data.corr()['price_range'].sort_values(ascending=False)
    st.bar_chart(corr_with_target)

# Scatter Plot - RAM vs Battery with Price Range as Color
if not train_data.empty and st.checkbox("Show RAM vs Battery with Price Range"):
    st.subheader("RAM vs Battery Capacity Colored by Price Range")
    fig_scatter = px.scatter(train_data, x='ram', y='battery_power', color='price_range', title="RAM vs Battery Capacity")
    st.plotly_chart(fig_scatter)

# Distribution of Numeric Features
if not train_data.empty and st.checkbox("Show Distribution of Numeric Features"):
    numeric_features = train_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_feature = st.selectbox("Select a Feature to Visualize", numeric_features)
    plt.figure(figsize=(8, 4))
    sns.histplot(train_data[selected_feature], kde=True)
    st.pyplot(plt)
    plt.close()

# Show price range distribution as a bar chart
if not train_data.empty:
    st.subheader("Price Range Distribution")
    st.bar_chart(train_data['price_range'].value_counts())

# Information when no datasets or model are available
if train_data.empty:
    st.write("Training dataset is empty.")
if model is None:
    st.write("No model loaded for predictions.")
