import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can also try "distilgpt2" for a smaller model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate responses using the GPT-2 model
def generate_response(prompt, max_length=100):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate a response using the model
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    
    # Decode and return the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Streamlit application layout
st.title("Islamic Banking Chatbot")
st.write("Ask me anything about Islamic banking!")

# Text input for the user
user_input = st.text_input("You: ", "")

if st.button("Get Response"):
    if user_input:
        # Create a prompt for the chatbot
        prompt = f"You are an expert in Islamic banking. A user asks: {user_input}\nAnswer comprehensively:"
        # Generate the chatbot's response
        response = generate_response(prompt)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please enter a question.")
