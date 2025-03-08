import joblib
import streamlit as st

models = {
    "SVM Binary": joblib.load("/home/teddy/Music/DS/model_app/svm_binary.pkl"),
    "SVM Multi": joblib.load("/home/teddy/Music/DS/model_app/svm_multi.pkl"),
    "Logistic Binary": joblib.load("/home/teddy/Music/DS/model_app/logistics_binary.pkl"),
    "Logistic Multi (OVR)": joblib.load("/home/teddy/Music/DS/model_app/logistics_ovr.pkl"),
    "Logistic Multi (Multinomial)": joblib.load("/home/teddy/Music/DS/model_app/logistics_multinomial.pkl")
}

st.title("Multi Model Prediction")

selected_model = st.selectbox("Choose a model", list(models.keys()))

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1, value=3.0)

prediction = None
if st.button("Predict"):
    model = models[selected_model]
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(user_input)[0]
if prediction is not None:
    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"The model predicts: {class_names[prediction]}")