**Data Analysis on Mobile Price Classification**

Project Description

This project uses machine learning techniques to classify mobile phones into price ranges based on their features. It includes data preprocessing, implementation of various classification algorithms, and deployment of a Streamlit-based web application for real-time predictions.

## Live Demo  

👉 Check out the live project here: [data-analysis-on-mobile-price.onrender.com](https://data-analysis-on-mobile-price.onrender.com/)


**Features**

Jupyter Notebook: Includes detailed data preprocessing, exploratory data analysis (EDA), and model training steps.

PKL File: Serialized model for easy deployment.

Streamlit Application: Interactive web interface for real-time mobile price predictions.

**Installation**

Clone the repository:

git clone https://github.com/JayGemawat/Data-Analysis-on-Mobile-Price-Classification.git)
cd mobile-price-classification

Install required dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

**Usage**

Open the Streamlit app in your browser.

Enter mobile phone specifications in the provided fields.

View the predicted price range and model confidence.

**File Structure**

notebooks/: Contains the main Jupyter Notebook (data_analysis_mobile_price_classification.ipynb).

models/: Contains the trained model file (model.pkl).

app.py: Streamlit application for predictions.

requirements.txt: List of required Python libraries.

**Models Used**

The following machine learning models were implemented and evaluated:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Trees

Random Forest

Voting Classifier

AdaBoost

**Results**

The Voting Classifier achieved the highest accuracy of 96%.

Feature importance analysis showed that RAM and battery capacity were the most influential features.

**License**

This project is licensed under the MIT License - see the LICENSE file for details.

**Contributors**

Diya Patel

Jay Gemawat

**Acknowledgments**

Dataset sourced from Kaggle.

Streamlit for providing an interactive web development framework.

