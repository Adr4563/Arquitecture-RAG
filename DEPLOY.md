# Despliegue: PC (backend) + Raspberry Pi (frontend)

Backend (Ollama + `embed_server.py`) corre en la PC, `chat.py` corre acá.

Antes de correr `chat.py`, exportar:

```bash
export EMBED_SERVER_HOST=http://192.168.1.44:8081
export CHAT_SERVER_HOST=http://192.168.1.44:11434
```

(Metelas en un `.env`/perfil de shell si querés que persista entre sesiones.)

La IP puede cambiar si el router reasigna DHCP — si deja de conectar,
revisar `ipconfig` en la PC para la IP actual.

## Verificar la conexión antes de correr `chat.py`

```bash
curl -m 5 http://192.168.1.44:11434/api/tags
curl -m 5 -X POST http://192.168.1.44:8081/pregunta -H "Content-Type: application/json" -d '{"query":"test","n_results":1}'
```

Si ambos responden 200, seguí con `pip install -r requirements.txt` (o instalá
`requests`/`httpx` a mano) y probá `python chat.py`.

Si `curl` no conecta, antes de tocar código revisá que la Raspberry Pi esté en
la misma red que la PC — eso no lo arregla el código, es de red.

## Carita en la pantalla HDMI (frontend/display.py)

Con la Pi headless (sin sesión gráfica) y una pantalla LCD/HDMI conectada
directo, `display.py` detecta eso automáticamente y muestra la carita vía
`mpv --vo=drm` (dibuja directo por DRM/KMS, sin necesitar X11/Wayland) en vez
del visor Tkinter que usa la demo en PC.

Requisitos una sola vez en la Pi:

```bash
sudo apt install -y mpv
sudo usermod -aG video,render $USER
```

El grupo recién aplica en una sesión nueva — cerrá y volvé a conectarte por
SSH (o `newgrp video` para probarlo sin salir de la sesión actual).
