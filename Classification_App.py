import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

st.set_page_config(layout="wide")

## --Helper Functions--

# Load MLflow models
@st.cache_resource
def load_mlflow_models():
    try:
        # Set MLflow experiment
        mlflow.set_experiment("Text Classification")
        
        # Get the experiment
        experiment = mlflow.get_experiment_by_name("Text Classification")

        if experiment is None:
            st.error("MLflow experiment not found!")
            return None, None
        
        # Load the latest runs for each model
        models = {}
        metrics = {}
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"]:
            model_runs = runs[runs["params.model"] == model_name]
            if not model_runs.empty:
                latest_run = model_runs.iloc[0]
                run_id = latest_run.run_id
                
                # Load the model
                models[model_name] = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

                # Store metrics
                metrics[model_name] = {
                    "accuracy": latest_run["metrics.accuracy"],
                    "precision": latest_run["metrics.precision"],
                    "recall": latest_run["metrics.recall"],
                    "f1_score": latest_run["metrics.f1_score"],
                    "runtime": latest_run["metrics.runtime"]
                }
        
        return models, metrics
    except Exception as e:
        st.error(f"Error loading MLflow models: {str(e)}")
        return None, None


# Load label encoder
@st.cache_resource
def load_label_encoder():
    try:
        with open('label_encoder.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading label encoder: {str(e)}")
        return None

# Preprocess input text
def preprocess_text(text, text_columns):
    # Create a DataFrame with the same structure as training data
    input_df = pd.DataFrame(columns=text_columns)
    # Fill all columns with the input text
    for col in text_columns:
        input_df[col] = [text]
    return input_df

# Load models and encoder
models, metrics = load_mlflow_models()
label_encoder = load_label_encoder()


#  Selectbox options
options = ["Team Info", "Project Overview", "Performance Analysis", "Prediction"]

# Sidebar selectbox
with st.sidebar:
    selected_option = st.selectbox("Choose an option", options)

# --Content based on selected option--

if selected_option == "Team Info":
    st.title("Team Information")
    st.write("We are a team of data science enthusiasts passionate about leveraging machine learning for real-world applications.")
    st.write("Team Members:")
    st.write("- Member 1: [Kennety Mashishi] - [Team lead]")
    st.write("- Member 2: [Sarah Mahlangu] - [Project Manager]")
    st.write("- Member 3: [Gabamoitse Keefelakae] - [Data Scientist]")
    st.write("- Member 3: [Busisiwe Mbewe] - [Data Scientist]")

elif selected_option == "Project Overview":
    st.title("Project Overview")
    st.write("This project aims to develop a news article classification system using multiple machine learning models.  The goal is to accurately categorize news articles into predefined categories, providing a valuable tool for information organization, analysis, and retrieval.")
    st.write("Key Features:")
    st.write("- Multiple classification models: [Logistic Regression, Decision Tree, Random Forest, Naive Bayes]")
    st.write("- User-friendly Streamlit interface")
    st.write("- Real-time prediction on user-inputted articles")
    st.write("- [Techniques used, NLP pre-processing, Classification Modelling]")

elif selected_option == "Performance Analysis":
    st.title("Model Performance Analysis")
    
    if metrics:
        # Performance comparison dataframe
        performance_df = pd.DataFrame(metrics).T
        
        # Metrics visualization
        st.subheader("Model Metrics Comparison")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Performance Metrics")
        
        # Plot metrics
        metrics_to_plot = [
            ("accuracy", "Accuracy", 0, 0),
            ("precision", "Precision", 0, 1),
            ("recall", "Recall", 1, 0),
            ("f1_score", "F1 Score", 1, 1)
        ]
        
        for metric, title, i, j in metrics_to_plot:
            performance_df[metric].plot(kind='bar', ax=axes[i,j])
            axes[i,j].set_title(title)
            axes[i,j].set_ylim(0, 1)
            axes[i,j].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Runtime comparison
        st.subheader("Model Runtime Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        performance_df['runtime'].plot(kind='bar')
        plt.title("Model Runtime Comparison")
        plt.ylabel("Runtime (seconds)")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Detailed metrics table
        st.subheader("Detailed Performance Metrics")
        st.dataframe(performance_df)

elif selected_option == "Prediction":
    st.title("News Article Classification")
    
    if models and label_encoder:
        text_input = st.text_area("Enter news article text:", height=200)
        
        if st.button("Classify"):
            if text_input:
                # Preprocess input
                text_columns = ['headlines', 'description', 'content', 'url']
                input_data = preprocess_text(text_input, text_columns)
                
                # Make predictions with all models
                results = []
                for name, model in models.items():
                    start_time = time.time()
                    prediction = model.predict(input_data)[0]
                    runtime = time.time() - start_time
                    
                    try:
                        proba = np.max(model.predict_proba(input_data)[0]) * 100
                    except:
                        proba = None
                    
                    results.append({
                        'Model': name,
                        'Predicted Category': label_encoder.inverse_transform([prediction])[0],
                        'Confidence': f"{proba:.2f}%" if proba is not None else "N/A",
                        'Runtime': f"{runtime:.4f} sec"
                    })
                
                # Display results
                results_df = pd.DataFrame(results)
                st.subheader("Classification Results:")
                st.dataframe(results_df)
                
                # Visualize predictions
                st.subheader("Model Predictions Visualization")
                fig, ax = plt.subplots(figsize=(10, 5))
                probas = [float(r['Confidence'].replace('%', '')) for r in results if r['Confidence'] != 'N/A']
                model_names = [r['Model'] for r in results if r['Confidence'] != 'N/A']
                if probas:
                    sns.barplot(x=model_names, y=probas)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Confidence (%)')
                    st.pyplot(fig)
            else:
                st.warning("Please enter text to classify.")
