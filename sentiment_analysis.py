import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_objects():
    # Load the CountVectorizer
    with open('Models/countVectorizer.pkl', 'rb') as file:
        cv = pickle.load(file)
    
    # Load the MinMaxScaler
    with open('Models/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    # Load the Random Forest model
    with open('Models/random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    return cv, scaler, model

# Preprocess the input text
def preprocess_text(text):
    # Initialize PorterStemmer and stopwords
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
    
    # Remove non-alphabetic characters
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase and split into words
    text = text.lower().split()
    
    # Remove stopwords and apply stemming
    text = [stemmer.stem(word) for word in text if word not in STOPWORDS]
    
    # Join the words back into a single string
    return ' '.join(text)

# Main function for the Streamlit app
def main():
    st.title("Amazon Alexa Review Sentiment Analysis")
    st.write("This app predicts whether a review is positive (1) or negative (0) using a trained Random Forest model.")
    
    # Load the model and preprocessing objects
    cv, scaler, model = load_model_and_objects()
    
    # Input text box for user review
    user_input = st.text_area("Enter your review here:")
    
    if st.button("Predict"):
        if user_input:
            # Preprocess the input text
            processed_text = preprocess_text(user_input)
            
            # Convert the text into a feature vector using CountVectorizer
            text_vector = cv.transform([processed_text]).toarray()
            
            # Scale the features using MinMaxScaler
            scaled_text_vector = scaler.transform(text_vector)
            
            # Make a prediction using the Random Forest model
            prediction = model.predict(scaled_text_vector)
            
            # Display the prediction
            if prediction[0] == 1:
                st.success("Prediction: Positive Feedback (1)")
            else:
                st.error("Prediction: Negative Feedback (0)")
        else:
            st.warning("Please enter a review to predict.")

# Run the app
if __name__ == "__main__":
    main()
