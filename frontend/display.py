"""
Controla la carita que se muestra en pantalla (frontend/faces/*.gif: happy,
sad, angry, content, speaking).

Vive en el frontend a propósito: la pantalla es "hardware conectado a la
máquina donde corre chat.py" (el Manager), no al servidor de embeddings/modelos
— en la versión final, la Raspberry Pi 4 con su LCD.

Por el momento (demo en PC) usa face_viewer.py, un visor propio en Tkinter
(viene incluido con Python, no depende de instalar nada aparte ni de tener
algo en el PATH). El visor se lanza UNA sola vez y se queda vivo durante toda
la sesión: cambiar de cara no relanza la ventana (eso competía por foco/
z-order con la terminal cada vez, y a veces se perdía esa carrera y la
ventana nueva quedaba tapada) — solo se reescribe qué gif tiene que mostrar,
vía un archivo de señal que el visor revisa periódicamente.

Cuando se pase a la LCD real de la Raspberry Pi, esto se puede volver a
cambiar (ej. por mpv en pantalla completa) sin tocar el resto de chat.py —
la interfaz (mostrar_cara/detener) no cambia.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(HERE, "faces")
VIEWER = os.path.join(HERE, "face_viewer.py")
ARCHIVO_SENAL = os.path.join(HERE, ".current_face")

CARAS_VALIDAS = {"happy", "sad", "angry", "content", "speaking"}

_proceso = None


def _asegurar_visor(ruta_gif_inicial):
    """Lanza la ventana del visor si todavía no hay una viva. No hace nada
    si ya hay una corriendo — ahí el cambio de cara va por el archivo de
    señal, no por relanzar el proceso."""
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        return
    _proceso = subprocess.Popen(
        [sys.executable, VIEWER, ruta_gif_inicial, ARCHIVO_SENAL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def mostrar_cara(nombre):
    """Le pide a la ventana del visor (ya abierta, o recién abierta si es la
    primera vez) que muestre faces/<nombre>.gif, en loop."""
    if nombre not in CARAS_VALIDAS:
        print(f"[display] Cara desconocida: {nombre!r} (válidas: {sorted(CARAS_VALIDAS)})")
        return

    ruta_gif = os.path.join(FACES_DIR, f"{nombre}.gif")
    if not os.path.isfile(ruta_gif):
        print(f"[display] No existe {ruta_gif}")
        return

    _asegurar_visor(ruta_gif)
    with open(ARCHIVO_SENAL, "w", encoding="utf-8") as f:
        f.write(ruta_gif)


def detener():
    """Cierra la ventana del visor, si hay una corriendo (ej. al salir del chat)."""
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        _proceso.terminate()
    _proceso = None
    try:
        os.remove(ARCHIVO_SENAL)
    except FileNotFoundError:
        pass
