"""
Chatbot Simple con Simulación de Contexto usando Groq
===================================================

Este archivo implementa un chatbot básico que demuestra conceptos fundamentales:
- Gestión de historial de conversación
- Simulación de contexto mediante acumulación de mensajes
- Interfaz simple con Streamlit
- Integración directa con la API de Groq

Diferencias con chatbot_gestionada.py:
- No usa LangChain (implementación más simple)
- Gestión manual del historial de conversación
- Menos funcionalidades pero más fácil de entender
- Ideal para entender los conceptos básicos

Tecnologías utilizadas:
- Streamlit: Interfaz web simple
- Groq: API directa para acceso a LLMs
- Python: Listas y diccionarios para gestión de memoria

Autor: Clase VI - CEIA LLMIAG
Propósito: Demostrar conceptos básicos de memoria conversacional

Instrucciones para ejecutar:
    streamlit run chatbot_simulacion_contexto.py

Requisitos:
    pip install streamlit groq

Variables de entorno necesarias:
    GROQ_API_KEY: Tu clave API de Groq
"""

# ========================================
# IMPORTACIÓN DE LIBRERÍAS
# ========================================

import streamlit as st    # Framework para interfaz web
import os                # Para acceder a variables de entorno
from groq import Groq    # Cliente directo de Groq (sin LangChain)

# ========================================
# CONFIGURACIÓN INICIAL Y AUTENTICACIÓN
# ========================================

# Obtener la clave API de Groq desde las variables de entorno
# Esta es una práctica de seguridad para no hardcodear credenciales
groq_api_key = os.environ.get("GROQ_API_KEY")

# Validar que la clave API esté disponible
if not groq_api_key:
    st.error("⚠️ GROQ_API_KEY no está configurada en las variables de entorno")
    st.info("💡 Configura tu clave API: export GROQ_API_KEY='tu-clave-aqui'")
    st.stop()

# Crear el cliente de Groq para comunicación directa con la API
# Nota: Aquí usamos el cliente nativo de Groq, no LangChain
try:
    client = Groq(api_key=groq_api_key)
    st.sidebar.success("✅ Cliente Groq conectado exitosamente")
except Exception as e:
    st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
    st.stop()

# ========================================
# GESTIÓN DE MEMORIA CONVERSACIONAL
# ========================================

# Inicializar el historial de conversación en el estado de la sesión de Streamlit
# st.session_state permite mantener datos entre ejecuciones de la aplicación
if "conversation_history" not in st.session_state:
    # Formato de lista de diccionarios compatible con la API de Groq
    # Cada mensaje tiene: {"role": "user"/"assistant", "content": "texto"}
    st.session_state.conversation_history = []
    st.sidebar.info("💬 Nueva conversación iniciada")
else:
    # Mostrar información del historial actual
    num_mensajes = len(st.session_state.conversation_history)
    st.sidebar.info(f"💬 Conversación activa: {num_mensajes} mensajes")

# Botón para limpiar el historial de conversación
if st.sidebar.button("🗑️ Limpiar Conversación"):
    st.session_state.conversation_history = []
    st.sidebar.success("✅ Conversación reiniciada")
    st.rerun()  # Recargar la aplicación


def generate_response(input_text):
    """
    Genera una respuesta del chatbot manteniendo el contexto de la conversación.
    
    Esta función demuestra la simulación básica de contexto:
    1. Agrega el mensaje del usuario al historial
    2. Envía TODO el historial al modelo (simulación de contexto)
    3. Recibe la respuesta del modelo
    4. Agrega la respuesta al historial
    5. Retorna la respuesta para mostrar
    
    Args:
        input_text (str): Texto ingresado por el usuario
        
    Returns:
        str: Respuesta generada por el modelo LLM
        
    Importante:
        - El contexto se simula enviando TODO el historial en cada llamada
        - Esto es diferente a tener "memoria real" como en LangChain
        - El modelo ve toda la conversación previa cada vez
    """
    
    try:
        # ========================================
        # PASO 1: AGREGAR MENSAJE DEL USUARIO
        # ========================================
        
        # Agregar el nuevo mensaje del usuario al historial
        # Formato: {"role": "user", "content": "mensaje del usuario"}
        st.session_state.conversation_history.append({
            "role": "user", 
            "content": input_text
        })
        
        # ========================================
        # PASO 2: PREPARAR CONTEXTO COMPLETO
        # ========================================
        
        # El historial completo se envía al modelo para simular contexto
        # Esto incluye todos los mensajes previos de user y assistant
        messages_for_api = st.session_state.conversation_history.copy()
        
        # Información de debug para estudiantes
        st.sidebar.caption(f"📤 Enviando {len(messages_for_api)} mensajes al modelo")
        
        # ========================================
        # PASO 3: LLAMADA A LA API DE GROQ
        # ========================================
        
        # Realizar la llamada a la API con el historial completo
        # El modelo LLaMA procesará toda la conversación para generar contexto
        chat_completion = client.chat.completions.create(
            messages=messages_for_api,           # Todo el historial de conversación
            model="llama3-8b-8192",             # Modelo Llama 3 con contexto de 8192 tokens
            temperature=0.7,                    # Controla la creatividad (0=determinista, 1=creativo)
            max_tokens=1000,                    # Máximo de tokens en la respuesta
            top_p=0.9,                         # Control de diversidad en la generación
        )
        
        # Extraer la respuesta del modelo
        response = chat_completion.choices[0].message.content
        
        # ========================================
        # PASO 4: AGREGAR RESPUESTA AL HISTORIAL
        # ========================================
        
        # Agregar la respuesta del assistant al historial para futuras referencias
        # Formato: {"role": "assistant", "content": "respuesta del modelo"}
        st.session_state.conversation_history.append({
            "role": "assistant", 
            "content": response
        })
        
        # Información adicional para el sidebar
        st.sidebar.success(f"✅ Respuesta generada ({len(response)} caracteres)")
        
        return response
        
    except Exception as e:
        # Manejo de errores con información educativa
        error_msg = f"Error al generar respuesta: {str(e)}"
        st.sidebar.error(f"❌ {error_msg}")
        
        # Remover el último mensaje del usuario si hubo error
        if st.session_state.conversation_history and \
           st.session_state.conversation_history[-1]["role"] == "user":
            st.session_state.conversation_history.pop()
        
        return f"Lo siento, ocurrió un error: {error_msg}"

# ========================================
# CONFIGURACIÓN DE LA INTERFAZ PRINCIPAL
# ========================================

# Configurar el título y descripción de la aplicación
st.title("🤖 Chatbot Simple con Simulación de Contexto")
st.markdown("""
**Ejemplo educativo de chatbot básico** que demuestra:
- 🧠 Simulación de contexto mediante historial completo
- 📝 Gestión manual de memoria conversacional
- 🔗 Integración directa con API de Groq (sin LangChain)
- 💡 Conceptos fundamentales de chatbots
""")

# Información del modelo en la barra lateral
st.sidebar.markdown("### 🧠 Configuración del Modelo")
st.sidebar.info("""
**Modelo**: Llama 3 (8B parámetros)
**Contexto**: 8,192 tokens
**Temperature**: 0.7 (equilibrado)
**Max Tokens**: 1,000
""")

# ========================================
# INTERFAZ DE ENTRADA DEL USUARIO
# ========================================

# Campo de entrada para el usuario
st.markdown("### 💬 Escribe tu mensaje:")
user_input = st.text_input(
    "Usuario:",
    placeholder="Ejemplo: Hola, ¿cómo estás? ¿De qué hablamos antes?",
    label_visibility="collapsed"
)

# Botón adicional para enviar
col1, col2 = st.columns([4, 1])
with col2:
    send_button = st.button("📤 Enviar", type="primary")

# ========================================
# PROCESAMIENTO Y VISUALIZACIÓN
# ========================================

# Procesar la entrada del usuario
if user_input and (user_input.strip() or send_button):
    # Mostrar indicador de carga
    with st.spinner('🤔 Generando respuesta...'):
        response = generate_response(user_input)
    
    # Mostrar la respuesta
    st.markdown("### 🤖 Respuesta del Chatbot:")
    st.markdown(f"""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
        {response}
    </div>
    """, unsafe_allow_html=True)

# ========================================
# MOSTRAR HISTORIAL DE CONVERSACIÓN
# ========================================

# Panel expandible con el historial completo
if st.session_state.conversation_history:
    with st.expander(f"📜 Ver Historial Completo ({len(st.session_state.conversation_history)} mensajes)"):
        for i, message in enumerate(st.session_state.conversation_history):
            role = "👤 Usuario" if message["role"] == "user" else "🤖 Chatbot"
            st.markdown(f"**{role}**: {message['content']}")
            if i < len(st.session_state.conversation_history) - 1:
                st.markdown("---")

# ========================================
# INFORMACIÓN EDUCATIVA PARA ESTUDIANTES
# ========================================

# Panel expandible con información técnica
with st.expander("📚 Información Educativa - Simulación de Contexto"):
    st.markdown("""
    ### ¿Cómo funciona la simulación de contexto?
    
    **1. Gestión Manual del Historial:**
    - Se mantiene una lista de todos los mensajes
    - Cada mensaje tiene formato: `{"role": "user/assistant", "content": "texto"}`
    - El historial se almacena en `st.session_state`
    
    **2. Simulación de Contexto:**
    - En cada consulta se envía TODO el historial al modelo
    - El modelo procesa toda la conversación previa
    - Esto simula "memoria" pero no es memoria real
    
    **3. Diferencias con LangChain:**
    - **Este enfoque**: API directa, implementación simple
    - **LangChain**: Gestión automática, más funcionalidades
    
    **4. Limitaciones de este Enfoque:**
    - Costo creciente (más tokens en cada llamada)
    - Límite de contexto del modelo (8,192 tokens)
    - No hay optimización de memoria
    
    **5. Ventajas Educativas:**
    - Fácil de entender
    - Control total sobre el proceso
    - Transparencia en el funcionamiento
    
    ### Estructura de Datos:
    ```python
    conversation_history = [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "¡Hola! ¿Cómo estás?"},
        {"role": "user", "content": "¿De qué hablamos?"},
        # ... más mensajes
    ]
    ```
    
    ### Flujo de Procesamiento:
    ```
    Usuario → Input → Agregar a historial → Enviar todo a API → 
    Respuesta → Agregar respuesta → Mostrar resultado
    ```
    """)

# ========================================
# PIE DE PÁGINA EDUCATIVO
# ========================================

st.markdown("---")
st.markdown("""
**📖 Clase VI - CEIA LLMIAG** | Ejemplo de simulación básica de contexto conversacional

💡 **Próximo paso**: Compara este enfoque con `chatbot_gestionada.py` para ver las diferencias con LangChain
""")

# Información de debug para desarrollo
if st.sidebar.checkbox("🔧 Modo Debug (para desarrolladores)"):
    st.sidebar.markdown("### Debug Info:")
    st.sidebar.json({
        "total_mensajes": len(st.session_state.conversation_history),
        "ultimo_mensaje": st.session_state.conversation_history[-1] if st.session_state.conversation_history else "Ninguno",
        "tokens_aproximados": sum(len(msg["content"]) for msg in st.session_state.conversation_history) // 4
    })
