from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import mlflow
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

st.set_page_config(layout="wide")

## --Helper Functions--

# Load MLflow models
# @st.cache_resource
# def load_mlflow_models():
#     try:
#         # Set MLflow experiment
#         mlflow.set_experiment("Text Classification")
        
#         # Get the experiment
#         experiment = mlflow.get_experiment_by_name("Text Classification")

#         if experiment is None:
#             st.error("MLflow experiment not found!")
#             return None, None
        
#         # Load the latest runs for each model
#         models = {}
#         metrics = {}
#         runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
#         for model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "Naive Bayes"]:
#             model_runs = runs[runs["params.model"] == model_name]
#             if not model_runs.empty:
#                 latest_run = model_runs.iloc[0]
#                 run_id = latest_run.run_id
                
#                 # Load the model
#                 models[model_name] = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

#                 # Store metrics
#                 metrics[model_name] = {
#                     "accuracy": latest_run["metrics.accuracy"],
#                     "precision": latest_run["metrics.precision"],
#                     "recall": latest_run["metrics.recall"],
#                     "f1_score": latest_run["metrics.f1_score"],
#                     "runtime": latest_run["metrics.runtime"]
#                 }

#                  # Print the metrics for debugging
#                 print(f"Model: {model_name}")
#                 print(f"  Accuracy: {metrics[model_name]['accuracy']}")
#                 print(f"  Precision: {metrics[model_name]['precision']}")
#                 print(f"  Recall: {metrics[model_name]['recall']}")
#                 print(f"  F1 Score: {metrics[model_name]['f1_score']}")
#                 print(f"  Runtime: {metrics[model_name]['runtime']}")
#                 print("-" * 40)
        
#         return models, metrics
#     except Exception as e:
#         st.error(f"Error loading MLflow models: {str(e)}")
#         return None, None


# @st.cache_resource
def load_models():
    try:
        models = {}
        
        metrics = {
            "Logistic Regression": {
                "accuracy": 0.978,
                "precision": 0.9784861279741949,
                "recall": 0.978,
                "f1_score": 0.978131806199115,
                "runtime": 4.088213682174683
            },
            # "Decision Tree": {
            #     "accuracy": 0.873,
            #     "precision": 0.8762745767120285,
            #     "recall": 0.873,
            #     "f1_score": 0.8735506368567483,
            #     "runtime": 5.061999082565308
            # },
            "Random Forest": {
                "accuracy": 0.949,
                "precision": 0.952060134867567,
                "recall": 0.949,
                "f1_score": 0.9495432050384706,
                "runtime": 11.193910837173462
            },
            "Naive Bayes": {
                "accuracy": 0.9655,
                "precision": 0.9660624542730797,
                "recall": 0.9655,
                "f1_score": 0.9654937353454095,
                "runtime": 1.1163816452026367
            }
        }

        model_files = {
            "Logistic Regression": "logistic_regression_pipeline.pkl",
            # "Decision Tree": "decision_tree_pipeline.pkl",
            "Random Forest": "random_forest_pipeline.pkl",
            "Naive Bayes": "naive_bayes_pipeline.pkl"
        }

        print("Model:")

        try:
            df = pd.read_csv("test.csv") 
            print("Shape of test_df:", df.shape)

            text_columns = ['headlines', 'description', 'content', 'url']
            y_test = df['category'].astype(str)  # Correctly get y_test

            X_test = df[text_columns]  # Select the text columns directly

        except FileNotFoundError:
            st.error("Data file not found. Please upload or provide the correct path.")
            return None, None
        except KeyError as e:  # Handle missing columns
            st.error(f"Column {e} not found in test data.")
            return None, None

        for model_name, file_path in model_files.items():
            try:
                with open(file_path, 'rb') as f:
                    models[model_name] = pickle.load(f)

                start_time = time.time()
                y_pred = models[model_name].predict(X_test)  

                y_pred = y_pred.astype(str)
                y_test = y_test.astype(str)

                print(f"Unique classes in y_pred for {model_name}: {np.unique(y_pred)}") 
                print(f"Unique classes in y_test: {np.unique(y_test)}") 

                
                # y_pred = y_pred.astype(str)
                runtime = time.time() - start_time

            except Exception as e:
                st.warning(f"Could not load or evaluate {model_name}: {str(e)}")
                continue

        print("All Metrics:", metrics)
        return models, metrics
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
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

# models, metrics = load_mlflow_models()
models, metrics = load_models()
label_encoder = load_label_encoder()


#  Selectbox options
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
        {"name": "Gaba Keefelakae", "role": "Data Scientist"},
        {"name": "Busisiwe Mbewe", "role": "Data Scientist"},
    ]

    for member in team_members:
        st.markdown(f"<div class='team-member'>{member['name']} <span class='role'>({member['role']})</span></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif selected_option == "Project Overview":
    st.markdown("<h1 class='centered-title'>Project Overview</h1>", unsafe_allow_html=True)

    # Styling
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
        }
        .project-overview-container {
            padding: 20px;
            border: 1px solid #eee;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
         .key-features-description {
            margin-bottom: 5px;
            list-style-type: none; 
            margin-left: 20px; 
            text-align: center;
        }
         .key-features-title {  
            text-align: center; 
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .key-feature {
            margin-bottom: 5px;
            list-style-type: none; 
            margin-left: 20px; 
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

 
    st.markdown("<div class='key-features-description'>This project aims to develop a news article classification system using multiple machine learning models. The goal is to accurately categorize news articles into predefined categories, providing a valuable tool for information organization, analysis, and retrieval.</div>", unsafe_allow_html=True) 

    st.markdown("<div class='key-features-title'>Key Features:</div>", unsafe_allow_html=True) 

    st.markdown("""
    <ul class = "key-feature">
        <li class = "key-feature">Multiple classification models: [Logistic Regression, Decision Tree, Random Forest, Naive Bayes].</li>
        <li class = "key-feature">User-friendly Streamlit interface.</li>
        <li class = "key-feature">Real-time prediction on user-inputted articles.</li>
        <li class = "key-feature">[Techniques used, NLP pre-processing, Classification Modelling].</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

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
