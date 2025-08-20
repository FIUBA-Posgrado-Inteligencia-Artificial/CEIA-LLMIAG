# 🚀 Servidor MCP Educativo

Servidor Model Context Protocol (MCP) para demostración educativa en el curso de **CEIA - Large Language Models**.

## 📋 Requisitos

- Python 3.8+
- Pip o uv para manejo de paquetes

## ⚡ Instalación Rápida

```bash
# Instalar dependencias
pip install -r requirements_mcp.txt

# Ejecutar servidor
python server_mcp.py
```

## 🔧 Uso del Servidor

### Iniciar el Servidor

```bash
python server_mcp.py
```

El servidor se ejecutará en: **http://127.0.0.1:8000/mcp**

### Detener el Servidor

Presiona `Ctrl+C` en la terminal donde está ejecutándose.

## 🛠️ Capacidades Disponibles

### **Tools (Herramientas)**
- `add(a, b)` - Suma dos números enteros
- `multiply(a, b)` - Multiplica dos números enteros  
- `power(base, exponent)` - Calcula potencias
- `factorial(n)` - Calcula factoriales

### **Resources (Recursos)**
- `greeting://{name}` - Saludo personalizado
- `info://server` - Información del servidor
- `docs://tools` - Documentación de herramientas

### **Prompts (Plantillas)**
- `greet_user(name, style)` - Saludos con diferentes estilos
- `math_problem(operation, difficulty)` - Problemas matemáticos
- `explain_concept(concept, audience)` - Explicaciones técnicas

## 🔌 Conectar un Cliente

### Ejemplo en Python

```python
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

async def test_client():
    async with streamablehttp_client("http://127.0.0.1:8000/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Usar una herramienta
            result = await session.call_tool("add", {"a": 5, "b": 3})
            print(f"5 + 3 = {result.content[0].text}")
            
            # Leer un recurso
            greeting = await session.read_resource(AnyUrl("greeting://Ana"))
            print(greeting.contents[0].text)
            
            # Obtener un prompt
            prompt = await session.get_prompt("greet_user", {"name": "Carlos", "style": "formal"})
            print(prompt.messages[0].content.text)

# Ejecutar cliente de prueba
asyncio.run(test_client())
```

### Desde Jupyter Notebook

Puedes usar el código del archivo `MCP_demo_rapido.ipynb` como referencia para conectarte desde un notebook.

## 📚 Estructura del Código

```
server_mcp.py          # Servidor MCP principal
requirements_mcp.txt   # Dependencias necesarias
README_MCP.md         # Esta documentación
ejemplo_MCP.py        # Ejemplo educativo completo
MCP_demo_rapido.ipynb # Demo en notebook
```

## 🎓 Para Estudiantes

### Experimentos Sugeridos

1. **Agregar nuevas tools**: Implementa `subtract`, `divide`, etc.
2. **Crear nuevos resources**: Agrega `status://health`, `data://students`  
3. **Diseñar prompts**: Crea templates para casos de uso específicos
4. **Manejo de errores**: Mejora la validación y manejo de excepciones
5. **Logging avanzado**: Implementa métricas y monitoreo

### Modificaciones del Código

El servidor está diseñado para ser fácilmente extensible:

```python
# Agregar nueva tool
@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Resta dos números enteros."""
    return a - b

# Agregar nuevo resource
@mcp.resource("status://health")
def get_health() -> str:
    """Estado de salud del servidor."""
    return "Server is running normally ✅"

# Agregar nuevo prompt
@mcp.prompt()
def code_review(language: str, complexity: str = "medium") -> str:
    """Prompt para revisión de código."""
    return f"Review this {language} code with {complexity} complexity level..."
```

## 🐛 Solución de Problemas

### Error: "FastMCP no está instalado"
```bash
pip install "mcp[cli]"
```

### Error: "Puerto 8000 en uso"
- Detén otros procesos que usen el puerto 8000
- O modifica el puerto en el código del servidor

### Error de conexión del cliente
- Verifica que el servidor esté ejecutándose
- Confirma la URL: `http://127.0.0.1:8000/mcp`

## 📖 Referencias

- [Documentación oficial MCP](https://modelcontextprotocol.io/)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Especificación técnica](https://spec.modelcontextprotocol.io/)

## 👨‍🏫 Autor

**CEIA - Especialización en Inteligencia Artificial**  
Curso: Large Language Models  
Año: 2024

---

💡 **¿Preguntas?** Consulta con tu profesor o revisa la documentación oficial de MCP.
