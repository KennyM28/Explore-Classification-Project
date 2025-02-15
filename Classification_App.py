import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

#  Selectbox options
options = ["Team Info", "Project Overview", "EDA", "Prediction"]

# Sidebar selectbox
with st.sidebar:
    selected_option = st.selectbox("Choose an option", options)

# --- Content based on selected option ---

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
    st.write("- Multiple classification models: [Logistic Regression, Decision Tree, Random Forest, SVM (RBF), Naive Bayes]")
    st.write("- User-friendly Streamlit interface")
    st.write("- Real-time prediction on user-inputted articles")
    st.write("- [Techniques used, NLP pre-processing, Classification Modelling]")

elif selected_option == "EDA":
    st.title("Exploratory Data Analysis")

    try:
        data = pd.read_csv('test.csv')  
        st.subheader("Sample Data")
        st.dataframe(data.head())

        st.subheader("Distribution of Categories")
        category_counts = data['category'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=category_counts.index, y=category_counts.values, ax=ax)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        # st.subheader("Article Image")
        # st.image("coverage.png")  

    except FileNotFoundError:
        st.error("Data file not found.")
    except Exception as e:
        st.error(f"An error occurred during EDA: {e}")

elif selected_option == "Prediction":
    st.subheader("News Classifier")
    st.write("Analysing news articles")

    st.markdown(
        """
        <div style="border: 1px solid #ADD8E6; padding: 10px; border-radius: 5px; background-color: #E0F2F7; text-color: black;"> 
            <p style="margin-bottom: 0;">Prediction with ML Models</p> 
        </div>
        """,
        unsafe_allow_html=True,
    )

    text_input = st.text_area("Enter Text", height=100)

    if st.button("Classify"):
        if text_input:
            # Load your models ONCE at the beginning of the script (see below)

            # Make predictions (replace with your actual prediction code)
            # Example:
            # prediction1 = model1.predict([text_input])[0]
            # ...

            st.write("Predictions:")
            # st.write(f"Model 1: {prediction1}") #... display for each model

            predictions_df = pd.DataFrame({
                "Model": ["Model 1", "Model 2", "Model 3"],  # Your model names
                "Prediction": ["Prediction 1", "Prediction 2", "Prediction 3"] # Your predictions
            })
            st.dataframe(predictions_df)

        else:
            st.warning("Please enter text to classify.")


# --- Model Loading (Do this ONCE at the start of your script) ---
# Example:
# import pickle
# try:
#     with open('model1.pkl', 'rb') as file:
#         model1 = pickle.load(file)
#     # ... Load other models
# except FileNotFoundError:
#     st.error("One or more model files not found.")
# except Exception as e:
#     st.error(f"An error occurred during model loading: {e}")