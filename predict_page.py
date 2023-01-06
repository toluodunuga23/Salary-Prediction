import streamlit as st
import pickle
import numpy as np

st.title("My Projects")
st.write("##")
st.write("##")
st.write("---")

st.subheader("Project 1: Salary Prediction")

st.write("##")

def load_model():
    with open("salary_model.pkl","rb") as file:
        data=pickle.load(file)
    return data

data= load_model()
 
regressor=data["model"]
le_country_loaded=data["le_country"]
le_education_loaded=data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("### We need some information to predict your salary")

    countries=(
        "United States of America",                             
        "Germany",                                               
        "United Kingdom of Great Britain and Northern Ireland", 
         "India",                                                   
        "Canada",                                                  
        "France",                                                  
        "Brazil",                                                  
        "Spain",                                                  
        "Netherlands",                                              
        "Australia",                                                
        "Italy",                                                   
        "Poland",                                                  
        "Sweden",                                                   
        "Russian Federation",                                       
        "Switzerland",
)

    education=(
        "Master's Degree", 
         "Bachelor's Degree", 
         'Other', 
        'Some College',
        'Professional', 
        'Doctorate', 
        'Elementary'
    )

    country= st.selectbox("Country", countries)
    education= st.selectbox("Education Level",education)
    experience= st.slider("Years of Experience",0,50,3)

    ok= st.button("Predict")
    if ok:
        X= np.array([[country,education,experience]])
        X[:,0]= le_country_loaded.transform(X[:,0])
        X[:,1]= le_education_loaded.transform(X[:,1])
        salary= regressor.predict(X)
        st.subheader(f"Your estimated salary is ${salary[0]:.2f}")
    