import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

model_path = 'model/model.pkl'

@st.cache(allow_output_mutation=True)
def load_model():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.sidebar.title("Informasi Model")
st.sidebar.write("Model menggunakan dataset B")
st.sidebar.write("Dibuat oleh: 2602113546- Muhammad Kaisar")
st.title("Prediksi Status")

st.write("Masukkan data input di bawah.")

gender = st.selectbox("Gender", ["Male", "Female"])
ssc_percentage = st.number_input("SSC Percentage")
hsc_percentage = st.number_input("HSC Percentage")
degree_percentage = st.number_input("Degree Percentage")
cgpa = st.number_input("CGPA")
entrance_exam_score = st.number_input("Entrance Exam Score")
technical_skill_score = st.number_input("Technical Skill Score")
soft_skill_score = st.number_input("Soft Skill Score")
internship_count = st.number_input("Internship Count")
live_projects = st.number_input("Live Projects")
work_experience_months = st.number_input("Work Experience (Months)")
certifications = st.number_input("Certifications")
attendance_percentage = st.number_input("Attendance Percentage")
backlogs = st.number_input("Backlogs")
extracurricular_activities = st.radio("Extracurricular Activities", ["Yes", "No"])
salary_package_lpa = st.number_input("Salary Package (LPA)")

gender_encoded = 1 if gender == "Male" else 0
extracurricular_activities_encoded = 1 if extracurricular_activities == "Yes" else 0

columns = ['gender', 'ssc_percentage', 'hsc_percentage', 'degree_percentage', 'cgpa', 'entrance_exam_score', 
           'technical_skill_score', 'soft_skill_score', 'internship_count', 'live_projects', 'work_experience_months', 
           'certifications', 'attendance_percentage', 'backlogs', 'extracurricular_activities', 'gender_encoded', 'salary_package_lpa']

input_data = np.array([[gender, ssc_percentage, hsc_percentage, degree_percentage, cgpa, entrance_exam_score, 
                        technical_skill_score, soft_skill_score, internship_count, live_projects, 
                        work_experience_months, certifications, attendance_percentage, backlogs, 
                        extracurricular_activities, gender_encoded, salary_package_lpa]])

input_data_df = pd.DataFrame(input_data, columns=columns)

if st.button("Prediksi"):
    prediction = model.predict(input_data_df)
    prediction_prob = model.predict_proba(input_data_df)

    
    st.success(f"Hasil Prediksi: {'Placed' if prediction[0] == 1 else 'Not Placed'} ({prediction_prob[0][np.argmax(prediction_prob)]*100:.2f}%)")
    st.subheader("Visualisasi Probabilitas")
    classes = ['Not Placed', 'Placed']
    probabilities = prediction_prob[0]
    
    fig, ax = plt.subplots()
    ax.bar(classes, probabilities, color=['green', 'yellow', 'red'])
    ax.set_ylabel('Probabilities')
    ax.set_title('Distribusi Probabilitas Prediksi')
    st.pyplot(fig)