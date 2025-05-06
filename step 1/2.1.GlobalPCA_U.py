# Task 2.1

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')


# === 1. Load the data ===
data = scipy.io.loadmat('CYLINDER_ALL.mat')
UALL = data['UALL']  # shape (np, nt)
nx = data["ny"][0][0]  # attention à l'inversion
ny = data["nx"][0][0]
nt_U = UALL.shape[1]

# === 2. Build spatial grid ===
x = np.linspace(-1, 8, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')  # (nx, ny)

# === 3. PCA Preparation ===

###########################################################################
#%% Streamwise component of the velocity, ux


X_U = UALL.T  # shape (nt_U, np)
scaler = StandardScaler()
X0_U = scaler.fit_transform(X_U)

# === 4. Apply PCA ===
pca = PCA()
pca.fit(X0_U)
Z_U = pca.transform(X0_U)
eigenvalues = pca.explained_variance_

# === 5. Plot Scree Plot ===
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(eigenvalues)+1), eigenvalues, marker='o')
plt.xlabel('Component index')
plt.ylabel('Eigenvalue Magnitude')
plt.title('Scree Plot - Streamwise Velocity (UALL)')
plt.grid(True)
plt.tight_layout()
plt.show()
# the plot shows us that PC = 6 is the best choice

# === 6. Plot PCA Projection (PC1 vs PC2) ===
plt.figure(figsize=(6, 6))
plt.scatter(Z_U[:, 2], Z_U[:, 3], c=np.arange(nt_U), cmap='viridis', s=50)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Global PCA Projection of Streamwise Velocity (UALL)')
plt.colorbar(label='Timestep')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 7. Reconstruct with q principal components ===
q = 6  # Number of components to keep
X0_rec_U = Z_U[:, :q] @ pca.components_[:q, :]
X_rec_U = scaler.inverse_transform(X0_rec_U)

# === 8. Plot original vs reconstructed at timestep t ===
t = 50  # Time snapshot to display
u_orig_U = UALL[:, t].reshape((nx, ny))       # original
u_rec_U  = X_rec_U[t, :].reshape((nx, ny))      # reconstructed

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
levels = np.linspace(-1.5, 1.5, 100)

cf1 = axes[0].contourf(X, Y, u_orig_U, levels=levels, cmap='viridis')
axes[0].set_title(f'Original $u_x$ at t={t}')
axes[0].set_xlabel('x / D')
axes[0].set_ylabel('y / D')

cf2 = axes[1].contourf(X, Y, u_rec_U, levels=levels, cmap='viridis')
axes[1].set_title(f'Reconstructed $u_x$ (q={q}) at t={t}')
axes[1].set_xlabel('x / D')
axes[1].set_ylabel('y / D')

plt.colorbar(cf1, ax=axes, orientation='vertical', shrink=0.8)
plt.tight_layout()
plt.show()

def reconstruction_error(X_original, X_reconstructed):
    num = np.linalg.norm(X_original - X_reconstructed, ord='fro')
    denom = np.linalg.norm(X_original, ord='fro')
    return (num / denom) * 100

err = reconstruction_error(X_U, X_rec_U)
print(f"Erreur relative de reconstruction (Local PCA): {err:.2f} %")
