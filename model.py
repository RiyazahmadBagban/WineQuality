

#pip install sreamlit

import streamlit as st
import pandas as pd
import pickle


st.title('Model Deployment on Wine-Quality dataset')



st.sidebar.header('User Input Parameters')




def user_input_features():
    fa = st.sidebar.number_input('fixed acidity',min_value=4.6,max_value=15.9,step=0.1)
    va = st.sidebar.number_input('volatile acidity',min_value=0.12,max_value=1.58,step=0.1)
    ca = st.sidebar.selectbox('citric acid',(0,1))

    rs = st.sidebar.number_input('residual sugar',min_value=0.9,max_value=15.5,step=0.1)
    ch = st.sidebar.number_input('chlorides',min_value=0.012,max_value=0.0611,step=0.001)
    fsd = st.sidebar.number_input('free sulfur dioxide',min_value=1,max_value=68,step=1)
    tsd = st.sidebar.number_input('total sulfur dioxide',min_value=6,max_value=289,step=1)
    dens = st.sidebar.number_input('density',min_value=0.99007,max_value=1.00369,step=0.00001)
    
    ph = st.sidebar.number_input('pH',min_value=2.74,max_value=4.01,step=0.01)
    sul = st.sidebar.number_input('sulphates',min_value=0.33,max_value=2.0,step=0.01)
    al = st.sidebar.number_input('alcohol',min_value=8.4,max_value=14.8,step=0.1)
    
    
    
   
    
    data= {'fixed acidity': fa,
         'volatile acidity': va,
         'citric acid': ca,
         'residual sugar': rs,
         'chlorides': ch,
         'free sulfur dioxide': fsd,
         'total sulfur dioxide': tsd,
         'density': dens,
         'pH': ph,
         'sulphates': sul,
         'alcohol': al}
    
    

    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)



with open(file="model.sav",mode="rb") as f1:
    model = pickle.load(f1)
    
    
prediction = model.predict(df)
#prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
#st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

#st.subheader('Prediction Probability')
st.write(prediction[0])



#live surver: streamlit.io, heroku, AWS, Azure














