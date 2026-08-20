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
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(HERE, "faces")

CARAS_VALIDAS = {"happy", "sad", "angry", "content", "speaking"}

# Ubicaciones típicas de instalación en Windows, por si "mpv" no está en el
# PATH de este proceso (ej. se instaló recién y la terminal/VSCode actual
# arrancó antes, con el PATH viejo en memoria — no vale la pena obligar al
# usuario a reiniciar todo solo por esto). En la Raspberry Pi ni se llega
# a mirar esta lista: shutil.which("mpv") ya lo encuentra vía apt.
_RUTAS_WINDOWS_FALLBACK = [
    r"C:\Program Files\MPV Player\mpv.exe",
    r"C:\Program Files (x86)\MPV Player\mpv.exe",
]


def _resolver_mpv():
    """Devuelve la ruta a usar para invocar mpv, o None si no se encuentra
    ni en el PATH ni en las ubicaciones típicas de Windows."""
    en_path = shutil.which("mpv")
    if en_path:
        return en_path
    if sys.platform == "win32":
        for ruta in _RUTAS_WINDOWS_FALLBACK:
            if os.path.isfile(ruta):
                return ruta
    return None


_proceso_actual = None
_mpv_disponible = True  # se apaga solo si no se encuentra en ningún lado, para no reintentar en cada turno


def mostrar_cara(nombre):
    """Reemplaza la carita en pantalla por faces/<nombre>.gif, en loop.

    Mata el reproductor anterior antes de lanzar el nuevo: son procesos de
    video en pantalla completa, no tiene sentido tener dos a la vez.
    """
    global _proceso_actual, _mpv_disponible

    if nombre not in CARAS_VALIDAS:
        print(f"[display] Cara desconocida: {nombre!r} (válidas: {sorted(CARAS_VALIDAS)})")
        return

    ruta_gif = os.path.join(FACES_DIR, f"{nombre}.gif")
    if not os.path.isfile(ruta_gif):
        print(f"[display] No existe {ruta_gif}")
        return

    if _proceso_actual is not None and _proceso_actual.poll() is None:
        _proceso_actual.terminate()

    if not _mpv_disponible:
        return

    mpv_bin = _resolver_mpv()
    if mpv_bin is None:
        _mpv_disponible = False
        print(f"[display] mpv no está instalado — mostrando '{nombre}' solo como texto. "
              "Instalar con: sudo apt install mpv (Pi) o winget install shinchiro.mpv (Windows)")
        return

    try:
        _proceso_actual = subprocess.Popen(
            [
                mpv_bin, "--loop", "--fullscreen", "--no-osc", "--no-input-terminal",
                "--no-input-default-bindings", "--really-quiet", ruta_gif,
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        _mpv_disponible = False
        print(f"[display] No se pudo ejecutar mpv en {mpv_bin!r} — mostrando '{nombre}' solo como texto.")


def detener():
    """Cierra el reproductor actual, si hay uno corriendo (ej. al salir del chat)."""
    global _proceso_actual
    if _proceso_actual is not None and _proceso_actual.poll() is None:
        _proceso_actual.terminate()
    _proceso_actual = None
