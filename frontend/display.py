"""
Controla la carita que se muestra en pantalla (frontend/faces/*.gif: happy,
sad, angry, content, speaking).

Vive en el frontend a propósito: la pantalla es "hardware conectado a la
máquina donde corre chat.py" (el Manager), no al servidor de embeddings/modelos
— en la versión final, la Raspberry Pi 4 con su LCD.

Por el momento (demo en PC) usa face_viewer.py, un visor propio en Tkinter
(viene incluido con Python, no depende de instalar nada aparte ni de tener
algo en el PATH). Cuando se pase a la LCD real de la Raspberry Pi, esto se
puede volver a cambiar por un reproductor a pantalla completa (mpv, etc.)
sin tocar el resto de chat.py — la interfaz (mostrar_cara/detener) no cambia.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(HERE, "faces")
VIEWER = os.path.join(HERE, "face_viewer.py")

CARAS_VALIDAS = {"happy", "sad", "angry", "content", "speaking"}

_proceso_actual = None


def mostrar_cara(nombre):
    """Reemplaza la carita en pantalla por faces/<nombre>.gif, en loop.

    Mata el visor anterior antes de lanzar el nuevo: no tiene sentido tener
    dos ventanas de cara a la vez.
    """
    global _proceso_actual

    if nombre not in CARAS_VALIDAS:
        print(f"[display] Cara desconocida: {nombre!r} (válidas: {sorted(CARAS_VALIDAS)})")
        return

    ruta_gif = os.path.join(FACES_DIR, f"{nombre}.gif")
    if not os.path.isfile(ruta_gif):
        print(f"[display] No existe {ruta_gif}")
        return

    if _proceso_actual is not None and _proceso_actual.poll() is None:
        _proceso_actual.terminate()

    # sys.executable: el mismo intérprete que ya está corriendo chat.py, así
    # que no hay que resolver ningún ejecutable externo por PATH.
    _proceso_actual = subprocess.Popen(
        [sys.executable, VIEWER, ruta_gif],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def detener():
    """Cierra el visor actual, si hay uno corriendo (ej. al salir del chat)."""
    global _proceso_actual
    if _proceso_actual is not None and _proceso_actual.poll() is None:
        _proceso_actual.terminate()
    _proceso_actual = None
