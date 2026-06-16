import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# Function to clean the text data
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    stop_words = stopwords.words('english')
    for i in text:
        if i.isalnum() and i not in stop_words:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ----------------- STREAMLIT UI ----------------- #

st.title("Email/SMS Spam Classifier 🚀")
st.write("Check if a message is spam or not using Machine Learning.")

# Text box for user input
input_sms = st.text_area("Enter your message here:")

# Action to perform when the button is clicked
if st.button('Predict'):
    
    if input_sms == "":
         st.warning("Please enter a message first!")
    else:
        with st.spinner('Checking...'):
            # 1. Preprocess: Clean the input message
            transformed_sms = transform_text(input_sms)
            
            # 2. Vectorize: Convert text to numbers
            vector_input = tfidf.transform([transformed_sms]).toarray()
            
            # 3. Predict: Use the model to find if it is spam (1) or not (0)
            result = model.predict(vector_input)[0]
            
            # 4. Display: Show the result on the screen
            if result == 1:
                st.error("🚨 This is a Spam Message!")
            else:
                st.success("✅ Not Spam. It is safe.")