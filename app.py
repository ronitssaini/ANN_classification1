import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

#Load the model
model =tf.keras.models.load_model('model.keras')

# Load the scaler and encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

#streamlit app
st.title('Customer Churn Prediction')
st.write('This app predicts whether a customer will churn or not based on their information.')
st.write('Please enter the customer information below:')

# Input fields
geography=st.selectbox('Geography',encoder.categories_[0])
gender=st.selectbox('Gender',label_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card', [0, 1])
is_active_member=st.selectbox('Is Active Member', [0, 1])

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Geography': [geography],
    'Age': [age],
    'Gender' : [label_encoder.transform([gender])[0]],
    'EstimatedSalary': [estimated_salary],
    'Balance': [balance],
    'CreditScore': [credit_score],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member]
})

#one hot encoding
geo_encoded=encoder.transform(input_data[['Geography']]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=encoder.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data,geo_encoded_df],axis=1)
input_data.drop(['Geography'],axis=1,inplace=True)

input_data = input_data[scaler.feature_names_in_]
input_data_scaled=scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

if prediction_probability > 0.5:
    st.write('The customer is likely to churn.')
else:   
    st.write('The customer is likely to stay.')
st.write(f'Prediction probability: {prediction_probability:.2f}')
st.write('---')