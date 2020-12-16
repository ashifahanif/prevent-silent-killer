

from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
model = load_model('diab_risk_pred_final')


def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('diab_risk_img1.jpg')
    image_diab = Image.open('Prediabetes.jpg')
    st.image(image,use_column_width=True)
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))
    st.sidebar.info('This app is created to predict if you are having early diabetics or not:')
    st.sidebar.success('https://www.pycaret.org')
    st.sidebar.image(image_diab)
    st.title("Predicting Early Diabetics")
    if add_selectbox == 'Online':
        Age = st.number_input('Patient Age' , min_value=0, max_value=150, value=1)
        Gender = st.selectbox('gender',['Male','female'])
        Polyuria = st.selectbox('Polyuria passing abnormal amount of urine',['yes','no'])
        Polydipsia = st.selectbox('feeling thirsty or persistent dry mouth', ['yes','no'])
        Sudden_weight_loss = st.selectbox('sudden weight loss',['yes','no'])
        Weakness = st.selectbox('feeling fatigue or weakness',['yes','no'])
        Polyphagia = st.selectbox('excessive hunger',['yes','no'])
        Genital_thrush = st.selectbox('vaginal thrush or vaginal itching',['yes','no'])
        Visual_blurring = st.selectbox('blurred vision',['yes','no'])
        Itching = st.selectbox('Itching on skin',['yes','no'])
        Irritability = st.selectbox('overall iritation',['yes','no'])
        Delayed_healing = st.selectbox('delay in healing wounds',['yes','no'])
        Partial_paresis = st.selectbox('weakening of muscles',['yes','no'])
        Muscle_stiffness = st.selectbox('feeling tightness on muscles',['yes','no'])
        Alopecia = st.selectbox('patches of hair loss',['yes','no'])
        Obesity = st.selectbox('obesity',['yes','no'])
        output=""
        input_dict={'Age':Age,'Gender':Gender,'Polyuria':Polyuria,'Polydipsia':Polydipsia,'Sudden_weight_loss': Sudden_weight_loss,
                    'Weakness':Weakness,'Polyphagia':Polyphagia,'Genital_thrush':Genital_thrush,'Visual_blurring':Visual_blurring,
                    'Itching':Itching,'Irritability':Irritability,'Delayed_healing':Delayed_healing,'Partial_paresis':Partial_paresis,
                    'Muscle_stiffness':Muscle_stiffness,'Alopecia':Alopecia,'Obesity':Obesity}

        input_df = pd.DataFrame([input_dict])
        if st.button("Predict your diabetics"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        st.success('your chance of diabetics {}'.format(output))
    if add_selectbox == 'Batch':
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
def main():
    run()

if __name__ == "__main__":
  main()
