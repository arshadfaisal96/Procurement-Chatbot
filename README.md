# California State Purchases Analysis Chatbot

This project creates an interactive chatbot to analyze and answer questions related to California's large purchase data. The chatbot predicts intent and retrieves relevant information, like total spending, supplier details, item prices, and more, using a pre-trained logistic regression model and NLP-based entity extraction.

# Dataset
The dataset used for this project contains large purchase records by the state of California. You can download it from Kaggle:

# Large Purchases by the State of California - Dataset on Kaggle: 
Download the dataset and place it in the root directory of this project as purchase_orders.csv.
download link: https://www.kaggle.com/datasets/sohier/large-purchases-by-the-state-of-ca

# Project Structure
chatbot_app.py: Main chatbot application code for running with Streamlit.

model_training.ipynb: Jupyter notebook for data preprocessing, feature engineering, and model training.

logistic_regression_model.pkl: Pre-trained logistic regression model for intent prediction.

tfidf_vectorizer.pkl: TF-IDF vectorizer for text transformation.

README.md: Project documentation.
