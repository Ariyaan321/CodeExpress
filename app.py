import numpy as np
import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer



# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Load the SavedModel
model = tf.saved_model.load("saved_model/sentiment_model_tf")
model_fn = model.signatures['serving_default']


# Define a function to predict sentiment
def predict_sentiment(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True)
    
    # Extract input_ids and attention_mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Prepare inputs for the model
    inputs_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    # Predict the sentiment
    predictions = model_fn(**inputs_dict)['logits']
    
    # Extract the predicted class (sentiment)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Define class labels (based on the model's training)
    class_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']    
    
    # Return the predicted label
    label = class_labels[predicted_class[0]]
    if(label == 'sadness' or 'anger' or 'fear' or 'surprise'):
        return 'complaint'
    else:
        return 'appreciation/suggestion'        


# Streamlit app interface
st.title("Classify Your Tweets: Complaints or Compliments")

# Create an input field for text
input_text = st.text_area("Enter a sentence to analyze:")

# Add a button for prediction
if st.button("Classify"):
    # Call the predict_sentiment function and display the result
    if input_text:
        result = predict_sentiment(input_text)
        st.write(f"Predicted Sentiment: {result}")
    else:
        st.write("Please enter a valid sentence.")
