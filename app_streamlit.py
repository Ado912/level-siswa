import joblib
import streamlit as st
import pandas as pd

model=joblib.load("model_level.joblib")

st.title("Klasifikasi Level Coding")
st.markdown("klasifikasi coding berdasarkan fitur hours coding daily,preferred language,typing speed,import usage and oop usage")

hoursCodingDaily=st.slider('hours_coding_daily',1.0,24.0,9.0)
preferredLanguage=st.pills('preferred_language',['Python','C++','Java'],default='C++')
typingSpeed=st.slider('typing_speed',10.0,100.0,40.0)
importUsage=st.pills('import_usage',['Yes','No'],default='Yes')
importOOP=st.pills('import_oop',['Yes','No'],default='Yes')

if st.button('prediksi',type='primary'):
	data_baru=pd.DataFrame([[hoursCodingDaily,preferredLanguage,typingSpeed,importUsage,importOOP]],columns=['hours_coding_daily', 'preferred_language', 	'typing_speed','import_usage', 'oop_usage'])
	prediksi=model.predict(data_baru)[0]
	presentase=max(model.predict_proba(data_baru)[0])
	st.success(f"model memprediksi {prediksi} dengan presentase {presentase*100:.2f}%")
	st.snow()