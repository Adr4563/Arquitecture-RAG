"""
Agente verificador: revisa el texto que generó otro worker antes de
mostrárselo al usuario — para Chat libre y Búsqueda Web.

Trivia no lo necesita: ya tiene su propio veredicto objetivo en
corrector.py (evaluar_respuesta, acierto/error contra una respuesta_esperada
conocida). Acá no hay una "respuesta correcta" contra la cual comparar —
solo coherencia: que la respuesta generada de verdad conteste lo que se le
preguntó, sin desviarse a algo incoherente o fuera de tema.
"""

from llama_client import generar_respuesta


def verificar_y_corregir(pregunta_original, respuesta_generada):
    """Si la respuesta es coherente y responde lo que se pidió, se devuelve
    tal cual (sin reescribirla de más). Si está incoherente, fuera de tema,
    o inventa algo que no viene al caso, se pide una versión corregida."""
    mensajes = [
        {"role": "system", "content": (
            "Eres un revisor de calidad. Te dan una pregunta y la respuesta "
            "que generó otro asistente. Si la respuesta es coherente, "
            "responde a lo que se pidió y tiene sentido, repítela EXACTAMENTE "
            "igual, sin cambiar nada. Si está incoherente, fuera de tema, o "
            "inventa algo que no viene al caso, corrígela: da una versión "
            "breve y coherente en su lugar, manteniendo el mismo tono.\n"
            "Responde SOLO con el texto final (corregido o igual), nada más: "
            "sin explicar qué hiciste, sin comillas."
        )},
        {"role": "user", "content": (
            f"Pregunta: {pregunta_original}\n"
            f"Respuesta generada: {respuesta_generada}"
        )},
    ]
    # temperature baja: es una revisión, no una charla — poca variación entre corridas.
    return generar_respuesta(mensajes, temperature=0.1, max_tokens=150).strip()
