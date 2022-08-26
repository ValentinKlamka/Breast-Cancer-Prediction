import pickle
import streamlit as st
import pandas as pd
import xgboost

def load_pickle(model_path):
    model_opener = open(model_path, "rb")
    model = pickle.load(model_opener)
    model_opener.close()
    return model

def make_predictions(test_df, model):
    prediction = model.predict(test_df)
    return prediction

def generate_predictions(test_df):

    model_pickle_path = "src/model.pkl"
    model= load_pickle(model_pickle_path)
    prediction = make_predictions(test_df, model)
    return prediction

if __name__ == "__main__":
    st.error("Disclaimer: Don't use this for medical advice. This is just a prediction model to have fun and play around with. So have fun and play around. If you are looking for medical advice, please consult a doctor.")
    
    st.title("Breast Cancer Prediction")
    st.write("This is an app to predict whether a tumor is malignant or benign, given clinical data.")
    st.write("Please enter the following data:")
    clump_thickness = st.slider("Clump Thickness", 1, 10, 5)
    uniformity_cell_size = st.slider("Uniformity of Cell Size", 1, 10, 5)
    uniformity_cell_shape = st.slider("Uniformity of Cell Shape", 1, 10, 5)
    marginal_adhesion = st.slider("Marginal Adhesion", 1, 10, 5)
    single_epithelial_cell_size = st.slider("Single Epithelial Cell Size", 1, 10, 5)
    bare_nuclei = st.slider("Bare Nuclei", 1, 10, 5)
    bland_chromatin = st.slider("Bland Chromatin", 1, 10, 5)
    normal_nucleoli = st.slider("Normal Nucleoli", 1, 10, 5)
    mitoses = st.slider("Mitoses", 1, 10, 5)
    
    input_dict = {  "Clump Thickness": clump_thickness,
                    "Uniformity of Cell Size": uniformity_cell_size,
                    "Uniformity of Cell Shape": uniformity_cell_shape,
                    "Marginal Adhesion": marginal_adhesion,
                    "Single Epithelial Cell Size": single_epithelial_cell_size,
                    "Bare Nuclei": bare_nuclei,
                    "Bland Chromatin": bland_chromatin,
                    "Normal Nucleoli": normal_nucleoli,
                    "Mitoses": mitoses
                }
    input_data = pd.DataFrame([input_dict])

    if st.button("Predict!"):
        pred = generate_predictions(input_data)
        if bool(pred):
            st.error("Malign")
        else:
            st.success("Benign")

    st.info("The breast cancer dataset which this app is based on was obtained from the University of Wisconsin Hospitals, Madison from Dr. William H. Wolberg.")
