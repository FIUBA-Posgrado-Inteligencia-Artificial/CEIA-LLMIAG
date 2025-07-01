# iniciar con streamlit run chatbot_gestionada.py
import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os


def main():
    """
    Esta función es el punto de entrada principal de la aplicación. Configura el cliente de Groq, la interfaz de Streamlit y maneja la interacción del chat.
    """
    
    # Obtener la clave API de Groq
    groq_api_key = os.getenv('GROQ_API_KEY')  # Reemplaza 'your_api' con tu clave API real

    # El título y mensaje de bienvenida de la aplicación Streamlit
    st.title("Chat CEIA de ejemplo")
    st.write("¡Hola! Este es un ejemplo de chatbot con memoria persistente gestionada programáticamente con Langchain, utilizando Groq")

    # Agregar opciones de personalización en la barra lateral
    st.sidebar.title('Personalización')
    system_prompt = st.sidebar.text_input("Mensaje del sistema:")
    model = st.sidebar.selectbox(
        'Elige un modelo',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value = 5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="historial_chat", return_messages=True)

    user_question = st.text_input("Haz una pregunta:")

    # Variable de estado de la sesión
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat=[]
    else:
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},
                {'output': message['IA']}
            )


    # Inicializar el objeto de chat Groq con Langchain
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )


    # Si el usuario ha hecho una pregunta,
    if user_question:

        # Construir una plantilla de mensaje de chat utilizando varios componentes
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),  # Este es el mensaje del sistema persistente que siempre se incluye al inicio del chat.

                MessagesPlaceholder(
                    variable_name="historial_chat"
                ),  # Este marcador de posición será reemplazado por el historial de chat real durante la conversación. Ayuda a mantener el contexto.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # Esta plantilla es donde se inyectará la entrada actual del usuario en el mensaje.
            ]
        )

        # Crear una cadena de conversación utilizando el LLM (Modelo de Lenguaje) de LangChain
        conversation = LLMChain(
            llm=groq_chat,  # El objeto de chat Groq LangChain inicializado anteriormente.
            prompt=prompt,  # La plantilla de mensaje construida.
            verbose=True,   # Habilita la salida detallada, lo cual puede ser útil para depurar.
            memory=memory,  # El objeto de memoria conversacional que almacena y gestiona el historial de la conversación.
        )
        
        # La respuesta del chatbot se genera enviando el mensaje completo a la API de Groq.
        response = conversation.predict(human_input=user_question)
        message = {'humano': user_question, 'IA': response}
        st.session_state.historial_chat.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
