import google.generativeai as genai

genai.configure(api_key="AIzaSyBeWqpZGyi6y577gQI3FJgQ5_Z-6m_k9Xw")
model=genai.GenerativeModel(model_name="gemini-2.0-flash")
chat=model.start_chat(history=[])
while True:
    prompt=input()
    if(prompt=="exit"):
        break
    res=chat.send_message(prompt)
    print(res.text)