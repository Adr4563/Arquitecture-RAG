"""
Controla la carita que se muestra en la pantalla LCD conectada a la
Raspberry Pi 4 (frontend/faces/*.gif: happy, sad, angry, content, speaking).

Vive en el frontend a propósito: la LCD es hardware conectado a la máquina
donde corre chat.py (el Manager), no al servidor de embeddings/modelos.

Reproduce el gif en loop con mpv en pantalla completa. Si mpv no está
instalado (ej. corriendo esto en una PC de desarrollo sin pantalla, como al
probar chat.py en Windows), no revienta: avisa por consola y sigue en modo
texto nomás.

Requiere en la Raspberry Pi:
    sudo apt install mpv
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(HERE, "faces")

CARAS_VALIDAS = {"happy", "sad", "angry", "content", "speaking"}

_proceso_actual = None
_mpv_disponible = True  # se apaga solo tras el primer FileNotFoundError, para no reintentar en cada turno


def mostrar_cara(nombre):
    """Reemplaza la carita en pantalla por faces/<nombre>.gif, en loop.

    Mata el reproductor anterior antes de lanzar el nuevo: son procesos de
    video en pantalla completa, no tiene sentido tener dos a la vez.
    """
    global _proceso_actual, _mpv_disponible

    if nombre not in CARAS_VALIDAS:
        print(f"[display] Cara desconocida: {nombre!r} (válidas: {sorted(CARAS_VALIDAS)})")
        return

    ruta = os.path.join(FACES_DIR, f"{nombre}.gif")
    if not os.path.isfile(ruta):
        print(f"[display] No existe {ruta}")
        return

    if _proceso_actual is not None and _proceso_actual.poll() is None:
        _proceso_actual.terminate()

    if not _mpv_disponible:
        return

    try:
        _proceso_actual = subprocess.Popen(
            [
                "mpv", "--loop", "--fullscreen", "--no-osc", "--no-input-terminal",
                "--no-input-default-bindings", "--really-quiet", ruta,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        _mpv_disponible = False
        print(f"[display] mpv no está instalado — mostrando '{nombre}' solo como texto. "
              "Instalar con: sudo apt install mpv")


def detener():
    """Cierra el reproductor actual, si hay uno corriendo (ej. al salir del chat)."""
    global _proceso_actual
    if _proceso_actual is not None and _proceso_actual.poll() is None:
        _proceso_actual.terminate()
    _proceso_actual = None
