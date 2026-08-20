"""
Visor persistente de la carita animada (GIF) en una ventana Tkinter.

A diferencia de relanzar un proceso nuevo por cada cambio de cara (lento, y
cada ventana nueva tiene que competir por foco/z-order con la terminal — a
veces gana, a veces se queda tapada), este proceso arranca UNA sola vez y se
queda vivo: revisa cada cierto tiempo el archivo de señal y, si cambió, anima
ese gif nuevo en la misma ventana ya abierta y con foco.

Uso:
    python face_viewer.py <ruta_al_gif_inicial> <archivo_de_señal>
"""

import os
import sys
import tkinter as tk

POLL_MS = 150   # cada cuánto revisa si se pidió otra cara
FRAME_MS = 80   # delay entre frames de la animación (~12 fps)


class Visor:
    def __init__(self, ruta_inicial, archivo_senal):
        self.archivo_senal = archivo_senal
        self.ruta_actual = None
        self.frames = []
        self._anim_job = None

        self.root = tk.Tk()
        self.root.title("Cara del robot (demo)")
        self.root.configure(bg="black")

        ancho, alto = 480, 270
        x = (self.root.winfo_screenwidth() - ancho) // 2
        y = (self.root.winfo_screenheight() - alto) // 2
        self.root.geometry(f"{ancho}x{alto}+{x}+{y}")
        self.root.attributes("-topmost", True)
        self.root.lift()
        self.root.focus_force()

        self.label = tk.Label(self.root, bg="black")
        self.label.pack(expand=True, fill="both")

        self._cargar(ruta_inicial)
        self.root.after(POLL_MS, self._revisar_senal)

    def _cargar(self, ruta):
        if ruta == self.ruta_actual or not os.path.isfile(ruta):
            return

        frames = []
        i = 0
        while True:
            try:
                frames.append(tk.PhotoImage(file=ruta, format=f"gif -index {i}"))
            except tk.TclError:
                break
            i += 1
        if not frames:
            return

        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
        self.frames = frames
        self.ruta_actual = ruta
        self._animar(0)

    def _animar(self, idx):
        self.label.configure(image=self.frames[idx])
        self._anim_job = self.root.after(FRAME_MS, self._animar, (idx + 1) % len(self.frames))

    def _revisar_senal(self):
        try:
            with open(self.archivo_senal, encoding="utf-8") as f:
                ruta = f.read().strip()
            if ruta:
                self._cargar(ruta)
        except FileNotFoundError:
            pass
        self.root.after(POLL_MS, self._revisar_senal)

    def run(self):
        self.root.mainloop()


def main():
    if len(sys.argv) < 3:
        print("Uso: python face_viewer.py <ruta_al_gif_inicial> <archivo_de_señal>")
        sys.exit(1)
    Visor(sys.argv[1], sys.argv[2]).run()


if __name__ == "__main__":
    main()
