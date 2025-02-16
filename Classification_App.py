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
        
        # Debug: Check available columns
        st.write("MLflow runs columns:", runs.columns)
        
        if "params.model" not in runs.columns:
            st.error("Column 'params.model' not found in MLflow runs.")
            return None, None
        
        for model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"]:
            runs["params.model"] = runs["params.model"].astype(str)
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

# Selectbox options
options = ["Team Info", "Project Overview", "Performance Analysis", "Prediction"]

# Sidebar selectbox
with st.sidebar:
    selected_option = st.selectbox("Choose an option", options)

# --Content based on selected option--

if selected_option == "Team Info":
    st.markdown("<h1 class='centered-title'>Team Information</h1>", unsafe_allow_html=True)

    # Styling
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
        }
        .team-members-title {  
            text-align: center; 
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .team-members-description {
            text-align: center; 
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .team-member {
            text-align: center; 
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        .role {
            text-align: center; 
            font-style: italic;
            color: #777;
            margin-left: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='team-info-container'>", unsafe_allow_html=True)

    st.markdown("<div class='team-members-description'>We are a team of data science enthusiasts passionate about leveraging machine learning for real-world applications.</div>", unsafe_allow_html=True) 

    st.markdown("<div class='team-members-title'>Team Members and Roles:</div>", unsafe_allow_html=True) 

    team_members = [
        {"name": "Kennety Mashishi", "role": "Team Lead"},
        {"name": "Sarah Mahlangu", "role": "Project Manager"},
        {"name": "Gabamoitse Keefelakae", "role": "Data Scientist"},
        {"name": "Busisiwe Mbewe", "role": "Data Scientist"},
    ]

    for member in team_members:
        st.markdown(f"<div class='team-member'>{member['name']} <span class='role'>({member['role']})</span></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
