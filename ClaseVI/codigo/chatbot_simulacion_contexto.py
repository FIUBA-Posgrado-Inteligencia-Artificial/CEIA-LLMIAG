import streamlit as st
import os
from groq import Groq

# Carga la clave de API de GROQ desde las variables de entorno
groq_api_key = os.environ.get("GROQ_API_KEY")

# Crea el cliente de GROQ
client = Groq(
    api_key=groq_api_key,
)

# Inicializa el historial de conversación en el estado de la sesión
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

def generate_response(input_text):
    # Agrega el mensaje del usuario al historial de conversación
    st.session_state.conversation_history.append({"role": "user", "content": input_text})

    # Genera la respuesta del chatbot utilizando el modelo LLaMA 3 y el historial de la conversación
    chat_completion = client.chat.completions.create(
        messages=st.session_state.conversation_history,
        model="llama3-8b-8192",
    )
    response = chat_completion.choices[0].message.content

    # Agrega la respuesta del chatbot al historial de conversación
    st.session_state.conversation_history.append({"role": "assistant", "content": response})

    return response

# Configuración de la interfaz de Streamlit
st.title("Chatbot con LLaMA 3")
st.subheader("¡Hazme una pregunta!")

user_input = st.text_input("Usuario:", "")

if user_input:
    response = generate_response(user_input)
    st.write(f"**Chatbot**: {response}")
