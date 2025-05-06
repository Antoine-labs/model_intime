import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# === 1. Load Data ===
data = scipy.io.loadmat('CYLINDER_ALL.mat')
UALL = data['UALL']  # shape (np, nt)
nx = data["ny"][0][0]
ny = data["nx"][0][0]
nt = UALL.shape[1]

# Transpose pour avoir les snapshots ligne par ligne
X_U = UALL.T  # shape (nt, np)

# === 2. Standardize each feature (spatial point) ===
scaler = StandardScaler()
X0 = scaler.fit_transform(X_U)

# === 3. Clustering des snapshots ===
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X0)

# === 4. Local PCA dans chaque cluster ===
q = 6  # nombre de composantes principales
X0_rec_local = np.zeros_like(X0)

for k in range(optimal_k):
    mask = (cluster_labels == k)
    X_k = X0[mask]

    scaler_k = StandardScaler()
    X_k_scaled = scaler_k.fit_transform(X_k)

    pca_k = PCA()
    Z_k = pca_k.fit_transform(X_k_scaled)
    X_k_rec = scaler_k.inverse_transform(Z_k[:, :q] @ pca_k.components_[:q, :])

    X0_rec_local[mask] = X_k_rec

# === 5. Retour à l’espace non-standardisé
X_rec_local = scaler.inverse_transform(X0_rec_local)  # shape (nt, np)

# === 6. Visualisation d’une reconstruction ===
x = np.linspace(-1, 8, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')

t = 50  # timestep
u_orig = UALL[:, t].reshape((nx, ny))
u_rec = X_rec_local[t, :].reshape((nx, ny))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
levels = np.linspace(-1.5, 1.5, 100)

cf1 = axes[0].contourf(X, Y, u_orig, levels=levels, cmap='viridis')
axes[0].set_title(f'Original U (t={t})')
axes[0].set_xlabel('x / D')
axes[0].set_ylabel('y / D')

cf2 = axes[1].contourf(X, Y, u_rec, levels=levels, cmap='viridis')
axes[1].set_title(f'Reconstructed U (Local PCA, q={q}, t={t})')
axes[1].set_xlabel('x / D')
axes[1].set_ylabel('y / D')

plt.colorbar(cf1, ax=axes, orientation='vertical', shrink=0.8)
plt.tight_layout()
plt.show()


# === 7. Erreur de reconstruction ===
def reconstruction_error(X_original, X_reconstructed):
    num = np.linalg.norm(X_original - X_reconstructed, ord='fro')
    denom = np.linalg.norm(X_original, ord='fro')
    return (num / denom) * 100


err = reconstruction_error(X_U, X_rec_local)
print(f"Erreur relative de reconstruction (Local PCA): {err:.5f} %")
