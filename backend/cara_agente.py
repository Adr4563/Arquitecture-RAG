"""
Agente de caras: decide qué cara (happy/sad/angry/content) le corresponde
al texto que va a decir el asistente, según su tono — para Chat libre y
Búsqueda Web.

Trivia no lo usa: ahí la cara la decide corrector.elegir_cara() según si la
respuesta del usuario fue correcta o no (un criterio objetivo). Acá no hay
acierto/error que evaluar, así que el criterio es distinto: el tono de lo
que el asistente mismo está por decir.
"""

from llama_client import generar_respuesta

CARAS_POSIBLES = ["happy", "sad", "angry", "content"]


def elegir_cara_por_tono(texto):
    """Clasifica el tono de `texto` y devuelve la cara que le corresponde.
    Si el modelo devuelve algo fuera de las 4 válidas, cae en 'content'
    (tono neutral) en vez de romper."""
    mensajes = [
        {"role": "system", "content": (
            "Eres un clasificador de tono. Te dan una frase que va a decir un "
            "robot. Elige qué cara le corresponde según el tono:\n"
            "- happy: contenido positivo, buenas noticias, entusiasmo.\n"
            "- sad: contenido negativo, malas noticias, decepción.\n"
            "- angry: enojo, queja, frustración.\n"
            "- content: tono neutral/informativo, sin carga emocional marcada.\n"
            "Responde EXACTAMENTE con una palabra: happy, sad, angry o content."
        )},
        {"role": "user", "content": texto},
    ]
    # temperature=0: es clasificación, no charla — no queremos variación entre corridas.
    cara = generar_respuesta(mensajes, temperature=0, max_tokens=5).strip().lower()
    return cara if cara in CARAS_POSIBLES else "content"
