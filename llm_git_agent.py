# llm_git_api.py

import typer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from jinja2 import Template
import subprocess
from pathlib import Path
import chardet
import re
import uvicorn
from typing import Optional
import threading
import time
import json

app_typer = typer.Typer()
app_fastapi = FastAPI()

MAX_CHARS_PER_CHUNK = 30000

class DiffRequest(BaseModel):
    diff_content: str
    model: str = "deepseek-coder:instruct"
    
class CommitResponse(BaseModel):
    success: bool
    message: str
    title: str
    description: str
    error: Optional[str] = None

def load_prompt_template():
    try:
        with open("commit_prompt_es.txt", "r", encoding="utf-8") as f:
            return Template(f.read())
    except FileNotFoundError:
        default_template = """
Actúa como un desarrollador experto en Git y análisis de código. Tu tarea es analizar los cambios de código y generar mensajes de commit claros, técnicos y descriptivos en español.


INSTRUCCIONES:
- Ignora cambios triviales o irrelevantes como espacios, líneas en blanco, tabulaciones, estilo o formato que no afecten la lógica del código.
- Analiza cuidadosamente los cambios funcionales mostrados en el diff.
- Identifica las modificaciones clave, nuevas funcionalidades, correcciones de errores o refactorizaciones significativas.
- Usa un tono profesional y técnico, evitando jerga innecesaria.
- Asegúrate de que el mensaje sea comprensible para otros desarrolladores.
- Sigue las convenciones de commits comunes y asegúrate de que el mensaje sea claro y conciso.
- Si el archivo es un .designer.cs no lo analices, ya que es generado automáticamente por Visual Studio y no requiere un mensaje de commit y retorna un mensaje genérico.
- Identifica el propósito principal de los cambios.
- Genera un mensaje de commit conciso pero informativo.
- Usa terminología técnica apropiada y profesional.
- Sigue las mejores prácticas de commits convencionales.

FORMATO DE RESPUESTA REQUERIDO:
Responde EXACTAMENTE en este formato JSON sin comentarios adicionales:
{
  "title": "tipo: descripción concisa del cambio (máximo 72 caracteres)",
  "description": "Descripción detallada de los cambios realizados, explicando qué se modificó y por qué"
}

TIPOS DE COMMIT COMUNES:
- feat: nueva funcionalidad
- fix: corrección de errores
- refactor: refactorización de código
- docs: cambios en documentación
- style: cambios de formato/estilo
- test: agregar o modificar tests
- chore: tareas de mantenimiento

CONTEXTO DEL ANÁLISIS:
{% if total > 1 %}
Estás analizando la parte {{ parte }} de {{ total }} del diff completo.
{% else %}
Estás analizando el diff completo.
{% endif %}

DIFF A ANALIZAR:
```diff
{{ contenido }}
```

Responde únicamente con el JSON en el formato especificado.
"""
        return Template(default_template)

def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    for i in range(0, len(text), max_chars):
        yield text[i:i + max_chars]

def run_ollama(prompt: str, model: str):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            timeout=500  
        )
        if result.returncode != 0:
            raise Exception(f"Error ejecutando ollama: {result.stderr}")
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        raise Exception("Timeout ejecutando ollama")
    except Exception as e:
        raise Exception(f"Error ejecutando ollama: {str(e)}")

def parse_structured_response(response: str) -> tuple[str, str]:
    """
    Extrae título y descripción de la respuesta del modelo.
    Intenta parsear JSON primero, luego usa fallbacks.
    """
    try:
        # Limpiar la respuesta de posibles caracteres extra
        cleaned_response = response.strip()
        
        # Buscar JSON en la respuesta
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            title = parsed.get("title", "").strip()
            description = parsed.get("description", "").strip()
            
            if title and description:
                return title, description
    
    except (json.JSONDecodeError, KeyError):
        pass
    
    # Fallback: intentar extraer título y descripción manualmente
    lines = response.split('\n')
    title = ""
    description = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Buscar líneas que parezcan títulos (con tipo de commit)
        if re.match(r'^(feat|fix|refactor|docs|style|test|chore):', line, re.IGNORECASE):
            title = line[:72]  # Limitar a 72 caracteres
        elif line and not title:
            # Si no hay título aún, usar la primera línea no vacía
            # Intentar detectar el tipo de commit
            commit_type = detect_commit_type(line)
            title = f"{commit_type}: {line[:65]}"  # Dejar espacio para el tipo
        elif line and title and not description:
            description = line
            break
    
    # Si no se encontró descripción, usar el título como descripción
    if not description:
        description = title
    
    return title or "chore: actualización de código", description or "Cambios realizados en el código"

def detect_commit_type(text: str) -> str:
    """Detecta el tipo de commit basado en el contenido del texto"""
    text_lower = text.lower()
    
    if re.search(r'(correcci[oó]n|error|bug|fix|arregla)', text_lower):
        return "fix"
    elif re.search(r'(refactor|reorganiza|estructura|mejora)', text_lower):
        return "refactor"
    elif re.search(r'(doc|documentaci[oó]n)', text_lower):
        return "docs"
    elif re.search(r'(elimina|borra|remove|limpia)', text_lower):
        return "chore"
    elif re.search(r'(test|prueba)', text_lower):
        return "test"
    elif re.search(r'(estilo|formato|style)', text_lower):
        return "style"
    elif re.search(r'(a[gñ]ade|nueva|nuevo|implementa|crea)', text_lower):
        return "feat"
    else:
        return "feat"

def format_commit_message(title: str, description: str) -> tuple[str, str, str]:
    """
    Formatea el mensaje final del commit
    """
    # Asegurar que el título no exceda 72 caracteres
    if len(title) > 72:
        title = title[:72]
    
    # Crear el mensaje completo
    full_message = f"{title}\n\n{description}"
    
    return full_message, title, description

def summarize_diff(diff: str, model: str):
    if not diff.strip():
        raise Exception("El diff está vacío")
    
    prompt_template = load_prompt_template()
    summaries = []

    chunks = list(chunk_text(diff))
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks):
        prompt = prompt_template.render(parte=i + 1, total=total_chunks, contenido=chunk)
        summary = run_ollama(prompt, model)
        summaries.append(summary)

    # Si hay múltiples chunks, combinar los resúmenes
        reduce_prompt = f"""
Actúa como un desarrollador experto. A continuación tienes los resúmenes parciales de los cambios detectados en una rama. 
Combina toda la información en un solo mensaje de commit estructurado.

Resúmenes parciales:
{chr(10).join([f"Parte {i+1}: {summary}" for i, summary in enumerate(summaries)])}

Responde EXACTAMENTE en este formato JSON:
{{
  "title": "tipo: descripción concisa del cambio principal (máximo 72 caracteres)",
  "description": "Descripción detallada combinando todos los cambios realizados"
}}

Solo responde con el JSON, sin explicaciones adicionales.
"""
        final_response = run_ollama(reduce_prompt, model)
   
    # Extraer título y descripción de la respuesta estructurada
    title, description = parse_structured_response(final_response)
    
    return format_commit_message(title, description)

@app_fastapi.post("/generate-commit", response_model=CommitResponse)
async def generate_commit(request: DiffRequest):
    try:
        if not request.diff_content.strip():
            raise HTTPException(status_code=400, detail="El diff no puede estar vacío")
        
        full_message, title, description = summarize_diff(request.diff_content, request.model)
        
        return CommitResponse(
            success=True,
            message=full_message,
            title=title,
            description=description
        )
    except Exception as e:
        return CommitResponse(
            success=False,
            message="",
            title="",
            description="",
            error=str(e)
        )

@app_fastapi.get("/health")
async def health_check():
    return {"status": "ok", "service": "llm-git-agent"}

@app_fastapi.get("/check-ollama")
async def check_ollama():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            models = result.stdout.strip()
            return {"available": True, "models": models}
        else:
            return {"available": False, "error": result.stderr}
    except Exception as e:
        return {"available": False, "error": str(e)}

# Comando para iniciar el servidor
@app_typer.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    reload: bool = False
):
    """
    Inicia el servidor API para generar commits con LLM
    """
    typer.echo(f"Iniciando servidor en http://{host}:{port}")
    typer.echo("Endpoints disponibles:")
    typer.echo("  POST /generate-commit - Genera mensaje de commit")
    typer.echo("  GET  /health - Estado del servicio")
    typer.echo("  GET  /check-ollama - Verifica disponibilidad de ollama")
    
    uvicorn.run(
        "llm_git_agent:app_fastapi",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

# Comando CLI original (mantener compatibilidad)
@app_typer.command()
def commit_llm(
    repo_path: str = ".",
    model: str = "mistral",
    commit: bool = True,
    push: bool = False,
    pr: bool = False,
):
    from git import Repo
    
    repo = Repo(repo_path)
    diff = repo.git.diff("HEAD")

    if not diff:
        typer.echo("No hay cambios para analizar.")
        raise typer.Exit()

    typer.echo("Analizando cambios con el modelo local...")
    try:
        final_message, title, description = summarize_diff(diff, model)
    except Exception as e:
        typer.echo(f"Error generando commit: {e}")
        raise typer.Exit(1)

    typer.echo("Mensaje generado:")
    typer.echo(f"Título: {title}")
    typer.echo(f"Descripción: {description}")
    typer.echo(f"\nMensaje completo:\n{final_message}")

    if commit:
        repo.git.add(all=True)
        repo.index.commit(final_message)
        typer.echo("Commit realizado.")

    if push:
        origin = repo.remote(name="origin")
        origin.push()
        typer.echo("Cambios enviados a remoto.")

    if pr:
        typer.echo("\nEste mensaje puede ser usado para una Pull Request:")
        typer.echo(f"\nTítulo sugerido: {title}")
        typer.echo(f"\nDescripción completa:\n{description}")

if __name__ == "__main__":
    app_typer()
