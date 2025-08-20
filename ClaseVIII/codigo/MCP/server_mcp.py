"""
SERVIDOR MCP - MODEL CONTEXT PROTOCOL
=====================================

Servidor MCP educativo usando FastMCP que demuestra las tres capacidades principales:
- Tools: Herramientas ejecutables (funciones)
- Resources: Recursos de datos accesibles 
- Prompts: Templates de prompts reutilizables

Basado en el SDK oficial de MCP para Python.

Uso:
    python server_mcp.py

El servidor se ejecuta en: http://127.0.0.1:8000/mcp

Autor: Ezequiel Guinsburg (ezequiel.guinsburg@gmail.com) CEIA - Curso de NLP2
Fecha: Agosto2025
"""

import asyncio
import logging
import signal
import socket
import sys
import time
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.error("FastMCP no está instalado. Ejecuta: pip install 'mcp[cli]'")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN DEL SERVIDOR MCP
# ============================================================================

# Crear instancia del servidor MCP
mcp = FastMCP("Demo-MCP-Server")

# ============================================================================
# DEFINICIÓN DE TOOLS (HERRAMIENTAS)
# ============================================================================

@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Suma dos enteros y devuelve el resultado.
    
    Args:
        a: Primer número entero
        b: Segundo número entero
    
    Returns:
        La suma de a + b
    """
    logger.info(f"🧮 Ejecutando sum: {a} + {b}")
    resultado = a + b
    logger.info(f"✅ Resultado: {resultado}")
    return resultado

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """
    Multiplica dos enteros y devuelve el resultado.
    
    Args:
        a: Primer número entero
        b: Segundo número entero
    
    Returns:
        El producto de a × b
    """
    logger.info(f"🧮 Ejecutando multiply: {a} × {b}")
    resultado = a * b
    logger.info(f"✅ Resultado: {resultado}")
    return resultado

@mcp.tool()
def power(base: int, exponent: int) -> int:
    """
    Calcula la potencia de un número.
    
    Args:
        base: Número base
        exponent: Exponente
    
    Returns:
        El resultado de base^exponent
    """
    logger.info(f"🧮 Ejecutando power: {base}^{exponent}")
    resultado = base ** exponent
    logger.info(f"✅ Resultado: {resultado}")
    return resultado

@mcp.tool()
def factorial(n: int) -> int:
    """
    Calcula el factorial de un número.
    
    Args:
        n: Número entero no negativo
    
    Returns:
        El factorial de n (n!)
    """
    if n < 0:
        raise ValueError("El factorial no está definido para números negativos")
    
    logger.info(f"🧮 Calculando factorial: {n}!")
    
    resultado = 1
    for i in range(1, n + 1):
        resultado *= i
    
    logger.info(f"✅ Resultado: {resultado}")
    return resultado

# ============================================================================
# DEFINICIÓN DE RESOURCES (RECURSOS)
# ============================================================================

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    Recurso de saludo personalizado.
    
    Args:
        name: Nombre de la persona a saludar
    
    Returns:
        Mensaje de saludo personalizado
    """
    logger.info(f"📚 Generando saludo para: {name}")
    saludo = f"¡Hola, {name}! Bienvenido/a al servidor MCP educativo. 🎓"
    logger.info(f"✅ Saludo generado")
    return saludo

@mcp.resource("info://server")
def get_server_info() -> str:
    """
    Información sobre el servidor MCP.
    
    Returns:
        Información detallada del servidor
    """
    logger.info("📚 Obteniendo información del servidor")
    
    info = """
    🚀 SERVIDOR MCP EDUCATIVO
    ========================
    
    Servidor: Demo-MCP-Server
    Versión: 1.0
    Puerto: 8000
    Transporte: streamable-http
    
    Capacidades disponibles:
    • Tools: add, multiply, power, factorial
    • Resources: greeting://{name}, info://server, docs://tools
    • Prompts: greet_user, math_problem, explain_concept
    
    Desarrollado para: CEIA - Curso de LLMs
    """
    
    return info.strip()

@mcp.resource("docs://tools")
def get_tools_documentation() -> str:
    """
    Documentación de las herramientas disponibles.
    
    Returns:
        Documentación completa de todas las tools
    """
    logger.info("📚 Generando documentación de tools")
    
    docs = """
    📖 DOCUMENTACIÓN DE HERRAMIENTAS
    ===============================
    
    1. add(a: int, b: int) -> int
       Suma dos números enteros.
       Ejemplo: add(5, 3) = 8
    
    2. multiply(a: int, b: int) -> int
       Multiplica dos números enteros.
       Ejemplo: multiply(4, 7) = 28
    
    3. power(base: int, exponent: int) -> int
       Calcula la potencia de un número.
       Ejemplo: power(2, 3) = 8
    
    4. factorial(n: int) -> int
       Calcula el factorial de un número.
       Ejemplo: factorial(5) = 120
    
    💡 Todas las herramientas incluyen logging detallado
       y manejo de errores básico.
    """
    
    return docs.strip()

# ============================================================================
# DEFINICIÓN DE PROMPTS (PLANTILLAS)
# ============================================================================

@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """
    Prompt reutilizable para saludar usuarios con diferentes estilos.
    
    Args:
        name: Nombre del usuario
        style: Estilo del saludo (friendly, formal, casual)
    
    Returns:
        Prompt personalizado para el saludo
    """
    logger.info(f"📝 Generando prompt de saludo para {name} (estilo: {style})")
    
    styles = {
        "friendly": "Por favor, escribe un saludo cálido y amigable",
        "formal": "Por favor, escribe un saludo formal y profesional",
        "casual": "Escribe un saludo casual y relajado",
        "academic": "Genera un saludo apropiado para el entorno académico"
    }
    
    base_prompt = styles.get(style, styles["friendly"])
    prompt = f"{base_prompt} para {name}. El contexto es un servidor MCP educativo."
    
    logger.info("✅ Prompt generado")
    return prompt

@mcp.prompt()
def math_problem(operation: str, difficulty: str = "medium") -> str:
    """
    Genera prompts para problemas matemáticos.
    
    Args:
        operation: Tipo de operación (add, multiply, power, factorial)
        difficulty: Nivel de dificultad (easy, medium, hard)
    
    Returns:
        Prompt para generar un problema matemático
    """
    logger.info(f"📝 Generando prompt de problema matemático: {operation} ({difficulty})")
    
    operations = {
        "add": "suma",
        "multiply": "multiplicación", 
        "power": "potenciación",
        "factorial": "factorial"
    }
    
    difficulties = {
        "easy": "números pequeños (1-10)",
        "medium": "números medianos (10-100)", 
        "hard": "números grandes (100-1000)"
    }
    
    op_name = operations.get(operation, "operación matemática")
    diff_desc = difficulties.get(difficulty, "dificultad media")
    
    prompt = f"""
    Genera un problema de {op_name} con {diff_desc}.
    
    El problema debe incluir:
    1. Enunciado claro y educativo
    2. Datos numéricos apropiados para el nivel
    3. Contexto real o aplicado
    4. Espacio para que el estudiante resuelva paso a paso
    
    Hazlo apropiado para estudiantes de postgrado en IA.
    """
    
    logger.info("✅ Prompt de problema matemático generado")
    return prompt.strip()

@mcp.prompt()
def explain_concept(concept: str, audience: str = "graduate") -> str:
    """
    Genera prompts para explicar conceptos técnicos.
    
    Args:
        concept: Concepto a explicar
        audience: Audiencia objetivo (undergraduate, graduate, expert)
    
    Returns:
        Prompt para explicar el concepto
    """
    logger.info(f"📝 Generando prompt explicativo para: {concept} (audiencia: {audience})")
    
    audiences = {
        "undergraduate": "estudiantes de grado",
        "graduate": "estudiantes de postgrado",
        "expert": "profesionales expertos"
    }
    
    audience_desc = audiences.get(audience, "estudiantes de postgrado")
    
    prompt = f"""
    Explica el concepto de "{concept}" para {audience_desc}.
    
    La explicación debe incluir:
    1. Definición clara y precisa
    2. Contexto y relevancia
    3. Ejemplos prácticos
    4. Conexiones con otros conceptos relacionados
    5. Aplicaciones en inteligencia artificial (si aplica)
    
    Usa un lenguaje técnico apropiado para el nivel de la audiencia.
    """
    
    logger.info("✅ Prompt explicativo generado")
    return prompt.strip()

# ============================================================================
# FUNCIONES DE CONTROL DEL SERVIDOR
# ============================================================================

def handle_shutdown(signum, frame):
    """Maneja la señal de cierre del servidor."""
    logger.info(f"🛑 Recibida señal {signum}. Cerrando servidor...")
    sys.exit(0)

async def run_server_async():
    """Ejecuta el servidor MCP de forma asíncrona."""
    try:
        logger.info("🚀 Iniciando servidor MCP...")
        logger.info("📡 Transport: streamable-http")
        logger.info("🌐 URL: http://127.0.0.1:8000/mcp")
        logger.info("📚 Tools disponibles: add, multiply, power, factorial")
        logger.info("📖 Resources disponibles: greeting://{name}, info://server, docs://tools")
        logger.info("📝 Prompts disponibles: greet_user, math_problem, explain_concept")
        logger.info("=" * 60)
        
        # Ejecutar el servidor (esto bloquea hasta que se cierre)
        await mcp.run_async(transport="streamable-http")
        
    except KeyboardInterrupt:
        logger.info("🛑 Cierre del servidor solicitado por el usuario")
    except Exception as e:
        logger.error(f"❌ Error en el servidor: {e}")
        raise
    finally:
        logger.info("👋 Servidor MCP cerrado")

def run_server():
    """Ejecuta el servidor MCP de forma síncrona."""
    try:
        logger.info("🚀 Iniciando servidor MCP...")
        logger.info("📡 Transport: streamable-http")
        logger.info("🌐 URL: http://127.0.0.1:8000/mcp")
        logger.info("📚 Tools disponibles: add, multiply, power, factorial")
        logger.info("📖 Resources disponibles: greeting://{name}, info://server, docs://tools")
        logger.info("📝 Prompts disponibles: greet_user, math_problem, explain_concept")
        logger.info("=" * 60)
        
        # Ejecutar el servidor (esto bloquea hasta que se cierre)
        mcp.run(transport="streamable-http")
        
    except KeyboardInterrupt:
        logger.info("🛑 Cierre del servidor solicitado por el usuario")
    except Exception as e:
        logger.error(f"❌ Error en el servidor: {e}")
        raise
    finally:
        logger.info("👋 Servidor MCP cerrado")

# ============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================================

def main():
    """Función principal del servidor MCP."""
    
    # Configurar manejo de señales para cierre limpio
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    print("=" * 60)
    print("🎓 SERVIDOR MCP EDUCATIVO - CEIA")
    print("=" * 60)
    print()
    print("📋 INSTRUCCIONES:")
    print("1. El servidor se ejecutará en http://127.0.0.1:8000/mcp")
    print("2. Usa Ctrl+C para detener el servidor")
    print("3. Conecta tu cliente MCP a la URL anterior")
    print("4. Prueba las herramientas, recursos y prompts disponibles")
    print()
    print("📚 Para más información sobre MCP:")
    print("   • https://modelcontextprotocol.io/")
    print("   • https://github.com/modelcontextprotocol")
    print()
    
    try:
        # Ejecutar servidor
        run_server()
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
