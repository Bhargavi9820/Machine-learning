import streamlit as st
import pandas as pd
import pickle
import numpy as np  # If you need to generate random data for demonstration purposes


def predict_quality(size, weight, sweetness, crunchiness, juiciness, ripeness, acidity):
    
    
    return np.random.choice([0, 1])


def main():
    st.title("Apple Quality Assessment")
    st.write("Bhargavi Muramkar")
    
    # User input for apple attributes
    Size = st.number_input("Size", min_value=0.0, max_value=1000.0, value=100.0)
    Weight = st.number_input("Weight", min_value=0.0, max_value=1000.0, value=200.0)
    Sweetness = st.slider("Sweetness", min_value=0, max_value=10, value=5)
    Crunchiness = st.slider("Crunchiness", min_value=0, max_value=10, value=5)
    Juiciness = st.slider("Juiciness", min_value=0, max_value=10, value=5)
    Ripeness = st.slider("Ripeness", min_value=0, max_value=10, value=5)
    Acidity = st.slider("Acidity", min_value=0, max_value=10, value=5)
    
    
    if st.button("Predict"):
        
        
        prediction = predict_quality(Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity)
        
        
        if prediction == 0:
            st.write("The apple quality is good.")
        else:
            st.write("The apple quality is bad.")

if __name__ == "__main__":
    main()
