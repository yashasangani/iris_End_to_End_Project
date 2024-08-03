# Import all necessary packages
import pandas as pd
import numpy as np 
import pickle
import streamlit as st

# Load preprocessor
with open("notebook/pipeline.pkl", "rb") as file1:
    pre = pickle.load(file1)

# Load the model
with open("notebook/model.pkl", "rb") as file2:
    model = pickle.load(file2)

# Preprocess the data
def predict_data(sep_len, sep_wid, pet_len, pet_wid):
    dct = {
        "sepal_length" :  [sep_len],
        "sepal_width" : [sep_wid],
        "petal_length" : [pet_len],
        "petal_width" : [pet_wid]
    }
    xnew = pd.DataFrame(dct)
    xnew_pre = pre.transform(xnew)
    pred = model.predict(xnew_pre)
    prob = model.predict_proba(xnew_pre)
    max_prob = np.max(prob)
    return pred, max_prob


# Run streamlit app
if __name__ == "__main__":
    st.set_page_config(page_title="Iris Project Yasha")
    st.title("Iris Project - Yasha Sangani")
    st.subheader("Please provide below inputs")
    # Take input from user
    sep_len = st.number_input("Sepal Length : ", min_value=0.00, step=0.01)
    sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
    pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
    pet_wid = st.number_input("Petal Width : ", min_value=0.00, step=0.01)
    # Button to predict output
    submit = st.button("Predict")
    # If submit button pressed
    if submit:
        pred, max_prob = predict_data(sep_len, sep_wid, pet_len, pet_wid)
        st.subheader("Model Response :")
        st.subheader(f"Predcition : {pred[0]}")
        st.subheader(f"Probability : {max_prob:.4f}")
        st.progress(max_prob)
