import streamlit as st
import pickle

with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼"
)

st.title("💼 Employee Salary Prediction")

st.write(
    "Predict employee salary using Machine Learning."
)

experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0
)

if st.button("Predict Salary"):

    prediction = model.predict([[experience]])

    salary = prediction[0]

    st.success(
        f"Predicted Annual Salary: ₹{salary:,.2f}"
    )

st.write("---")
st.caption("Built using Python and Machine Learning")