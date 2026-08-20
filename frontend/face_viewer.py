"""
Visor simple de una carita animada (GIF) en una ventana Tkinter, en loop.

Uso:
    python face_viewer.py <ruta_al_gif>

Reemplazo liviano de mpv para hacer demos rápido en la PC: no requiere
instalar nada aparte de Python (Tkinter viene incluido en la instalación
estándar). No es la versión final para la LCD de la Raspberry Pi — cuando
se pase a la pantalla real, este visor se puede cambiar por uno a pantalla
completa sin tocar display.py (solo la forma en que se dibuja el gif).
"""

import sys
import tkinter as tk


def main():
    if len(sys.argv) < 2:
        print("Uso: python face_viewer.py <ruta_al_gif>")
        sys.exit(1)
    ruta = sys.argv[1]

    root = tk.Tk()
    root.title("Cara del robot (demo)")
    root.configure(bg="black")

    # Sin esto, la ventana puede abrirse con un tamaño mínimo o quedar detrás
    # de otras ventanas sin que se note — para una demo tiene que saltar a la
    # vista sí o sí.
    ancho, alto = 480, 270
    x = (root.winfo_screenwidth() - ancho) // 2
    y = (root.winfo_screenheight() - alto) // 2
    root.geometry(f"{ancho}x{alto}+{x}+{y}")
    root.attributes("-topmost", True)
    root.lift()
    root.focus_force()

    label = tk.Label(root, bg="black")
    label.pack(expand=True, fill="both")

    # Tkinter no trae un decoder de gif con "cuántos frames tiene": hay que
    # ir pidiendo el índice N+1 hasta que truene, ahí se sabe que ya no hay más.
    frames = []
    i = 0
    while True:
        try:
            frames.append(tk.PhotoImage(file=ruta, format=f"gif -index {i}"))
        except tk.TclError:
            break
        i += 1

    if not frames:
        print(f"No se pudo leer {ruta}")
        sys.exit(1)

    DELAY_MS = 80  # ~12 fps, no viene el delay original de cada gif en esta lectura

    def animar(idx=0):
        label.configure(image=frames[idx])
        root.after(DELAY_MS, animar, (idx + 1) % len(frames))

    animar()
    root.mainloop()


if __name__ == "__main__":
    main()
