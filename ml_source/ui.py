import streamlit as st
import numpy as np
import joblib
import requests
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing import image
import google.generativeai as genai

genai.configure(api_key="AIzaSyBeWqpZGyi6y577gQI3FJgQ5_Z-6m_k9Xw")
model=genai.GenerativeModel(model_name="gemini-2.0-flash")
chat = model.start_chat(history=[])

ml=load_model("D:\\Covid19\\Data\\covid_pneu.h5")
class_=["Covid19","Normal","Pneumonia"]
st.title("Pneumonia/Covid Prediction")
u=st.file_uploader("Upload file",type=["jpg","jpeg","png"])
if u:
    img=Image.open(u).convert("RGB")
    img=img.resize((224,224))
    st.image(img,use_container_width=True)
    img=image.img_to_array(img)
    imag=np.expand_dims(img,axis=0)/255.0
    pred=ml.predict(imag)
    res=np.argmax(pred)
    st.write(class_[res])
    predicted_clas=class_[res]
    if predicted_clas=="Normal":
        st.success("You are healthy")
    else:
        st.sidebar.subheader(" Diagnosis ")
        Age=st.sidebar.slider("Age",0,80)
        Gn=st.sidebar.selectbox("Gender",["Male","Female"])
        fv=st.sidebar.selectbox("Fever (Yes/No)",["Yes","No"])
        cf=st.sidebar.selectbox("Cough (Yes/No)",["Yes","No"])
        ft=st.sidebar.selectbox("Fatigue (Yes/No)",["Yes","No"])
        brt=st.sidebar.selectbox("Breathlessness (Yes/No)",["Yes","No"])
        cm=st.sidebar.selectbox("Comorbidity (Yes/No)",["Yes","No"])
        sta=st.sidebar.selectbox("Stage",["mild","moderate","severe"])
        tp=st.sidebar.selectbox("Type",["viral","bacteria"])
        ts=st.sidebar.slider("Tumor_Size",0,5)

        gn_num=1 if Gn=="Male" else 0
        fv_num=1 if fv=="Yes" else 0
        cf_num=1 if cf=="Yes" else 0
        ft_num=1 if ft=="Yes" else 0
        brt_num=1 if brt=="Yes" else 0
        cm_num=1 if cm=="Yes" else 0
        sta_num=0 if sta=="mild" else 1 if sta=="moderate" else 2 
        tp_num=0 if tp=="viral" else 1

        data_inp={
            "Age":Age,
            "Gender":gn_num,
            "Fever":fv_num,
            "Cough":cf_num,
            "Fatigue":ft_num,
            "Breathlessness":brt_num,
            "Comorbidity":cm_num,
            "Stage":sta_num,
            "Type":tp_num,
            "Tumor_Size":ts
        }
        sub=st.sidebar.button("Predict")
        if sub:
            res=requests.post("http://127.0.0.1:8000/prd",json=data_inp)
            response=res.json()
            st.write(response["Prediction"])
            # if(response["Prediction"]>0.5):
st.markdown("---")
st.markdown("### 💬 Chat with Gemini")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous Q&A
for q, a in st.session_state.chat_history:
    st.markdown(f"**❓ You:** {q}")
    st.markdown(f"**🤖 Gemini:** {a}")
    st.markdown("---")

# Chat input using form (triggers refresh on submit)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("🗨️ Ask something...", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("Gemini is thinking..."):
        try:
            response = chat.send_message(user_input)
            st.session_state.chat_history.append((user_input, response.text))
        except Exception as e:
            st.session_state.chat_history.append((user_input, f"⚠️ Error: {str(e)}"))

ml = joblib.load("comment.pkl")
vec = joblib.load("vector.pkl")
sent = {1: "Positive", 0: "Neutral", -1: "Negative"}

if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

if st.button("📝 Give Feedback"):
    st.session_state.show_feedback = not st.session_state.show_feedback

if st.session_state.show_feedback:
    st.sidebar.header("Feedback")
    feedback_text = st.sidebar.text_area("💬 Share your thoughts")

    if st.sidebar.button("Submit"):
        if feedback_text.strip() != "":
            v = vec.transform([feedback_text])
            prd = ml.predict(v)
            fb = sent.get(prd[0])
            st.sidebar.success(f"Your feedback is **{fb}**. Thank you!")
        else:
            st.sidebar.warning("Please enter some feedback.")




