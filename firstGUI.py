#----------------------------
# Importa modulos
#----------------------------

import tkinter as tk

# Crear instancia
win=tk.Tk()

# Agrega titulo
win.title("Mi primer GUI")

tk.Label(win, text="Esta es mi primer etiqueta").pack()
tk.Button(win, text="Soy un boton").pack()


#win.resizable(False,False)

# Inicia el GUI/ crea el principal loop
win.mainloop()
