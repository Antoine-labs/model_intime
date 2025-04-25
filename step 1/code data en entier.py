# -*- coding: utf-8 -*-

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm  # <-- AJOUTER ÇA EN HAUT


# === 1. Load the .mat data ===
mat = scipy.io.loadmat('CYLINDER_ALL.mat')

nx = mat['nx'].item()
ny = mat['ny'].item()
nt = mat['UALL'].shape[1]  # number of time steps

x = np.linspace(-1, 8, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)

# === 2. Prepare the figure and initial plots ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ["Streamwise Velocity $u_x$", "Spanwise Velocity $u_y$", "Vorticity $\\omega$"]
cmaps = ['viridis', 'viridis', 'coolwarm']

levels_u = np.linspace(-1.5, 1.5, 50)
levels_v = np.linspace(-1.0, 1.0, 50)
levels_w = np.linspace(-10, 10, 50)
levels_list = [levels_u, levels_v, levels_w]

plt.tight_layout()

# === 3. Update function for the animation ===
def update(frame):
    u = mat['UALL'][:, frame].reshape((ny, nx))
    v = mat['VALL'][:, frame].reshape((ny, nx))
    w = mat['VORTALL'][:, frame].reshape((ny, nx))
    fields = [u, v, w]

    for i, ax in enumerate(axes):
        axes[i].cla()  # Clear previous contour
        axes[i].contourf(X, Y, fields[i], levels=levels_list[i], cmap=cmaps[i])
        axes[i].set_title(titles[i] + f" (t={frame})")
        axes[i].set_xlabel("x/D")
        axes[i].set_ylabel("y/D")



# === Animation avec tqdm ===
def update_with_progress(frame):
    tqdm.write(f"Frame {frame+1}/{nt}")  # ligne discrète à chaque étape
    return update(frame)

# === 4. Create the animation ===
ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

# === 5. Save the animation ===
ani.save('cylinder_flow.mp4', writer='ffmpeg', dpi=150)
ani.save('cylinder_flow.gif', writer='pillow', fps=10)

plt.show()
