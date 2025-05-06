import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === 1. Load Data ===
data = scipy.io.loadmat('CYLINDER_ALL.mat')
VALL = data['VALL']
nx = data["ny"][0][0]
ny = data["nx"][0][0]
nt = VALL.shape[1]

X_V = VALL.T  # (nt, np)


# === 2. Standardize data ===
scaler = StandardScaler()
X0_V = scaler.fit_transform(X_V)

#%% Streamwise component of the velocity, ux
# === 3. Paramètres VQPCA ===
k = 5  # nombre de clusters
q = 6  # nombre de PCs dans chaque cluster
max_iter = 10  # nombre d’itérations VQPCA

# === 4. Initialisation des clusters (aléatoire) ===
np.random.seed(0)
idx = np.random.randint(0, k, size=X0_V.shape[0])

# === 5. Boucle VQPCA ===
for iteration in range(max_iter):
    print(f"🌀 Iteration {iteration + 1}/{max_iter}")

    # Étape 1 : Appliquer PCA locale à chaque cluster
    cluster_models = {}
    cluster_means = {}
    cluster_scalers = {}
    X0_rec = np.zeros_like(X0_V)

    for i in range(k):
        cluster_points = X0_V[idx == i]
        if cluster_points.shape[0] < q:
            continue
        scaler_k = StandardScaler()
        cluster_scaled = scaler_k.fit_transform(cluster_points)

        pca_k = PCA(n_components=q)
        Z_k = pca_k.fit_transform(cluster_scaled)
        cluster_rec = pca_k.inverse_transform(Z_k)
        cluster_rec_unscaled = scaler_k.inverse_transform(cluster_rec)

        cluster_models[i] = (pca_k, scaler_k)
        cluster_means[i] = np.mean(cluster_points, axis=0)
        cluster_scalers[i] = scaler_k

    # Étape 2 : Réattribution des observations selon l’erreur de reconstruction
    new_idx = np.zeros_like(idx)
    for i in range(X0_V.shape[0]):
        errors = []
        for c in range(k):
            if c not in cluster_models:
                errors.append(np.inf)
                continue
            scaler_k = cluster_scalers[c]
            pca_k = cluster_models[c][0]
            X_scaled = scaler_k.transform(X0_V[i].reshape(1, -1))
            Z = X_scaled @ pca_k.components_.T
            X_rec_scaled = Z[:, :q] @ pca_k.components_[:q, :]
            X_rec = scaler_k.inverse_transform(X_rec_scaled)
            err = np.linalg.norm(X0_V[i] - X_rec.ravel())
            errors.append(err)
        new_idx[i] = np.argmin(errors)
    idx = new_idx

# === 6. Reconstruction finale avec VQPCA ===
X0_rec = np.zeros_like(X0_V)
for c in range(k):
    mask = (idx == c)
    if np.sum(mask) < q:
        continue
    X_k = X0_V[mask]
    scaler_k = StandardScaler()
    X_scaled = scaler_k.fit_transform(X_k)
    pca_k = PCA(n_components=q)
    Z_k = pca_k.fit_transform(X_scaled)
    X_rec_scaled = Z_k @ pca_k.components_
    X_rec = scaler_k.inverse_transform(X_rec_scaled)
    X0_rec[mask] = X_rec

X_rec = scaler.inverse_transform(X0_rec)

# === 7. Visualisation reconstruction (timestep fixe) ===
x = np.linspace(-1, 8, nx)
y = np.linspace(-2, 2, ny)
Xgrid, Ygrid = np.meshgrid(x, y, indexing='ij')
t = 50

u_orig = VALL[:, t].reshape((nx, ny))
u_rec = X_rec[t].reshape((nx, ny))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
levels = np.linspace(-1.5, 1.5, 100)

cf1 = axes[0].contourf(Xgrid, Ygrid, u_orig, levels=levels, cmap='viridis')
axes[0].set_title(f'Original U (t={t})')

cf2 = axes[1].contourf(Xgrid, Ygrid, u_rec, levels=levels, cmap='viridis')
axes[1].set_title(f'Reconstructed U (VQPCA, q={q})')

plt.colorbar(cf1, ax=axes, orientation='vertical')
plt.tight_layout()
plt.show()


# === 8. Erreur globale ===
def reconstruction_error(X_true, X_rec):
    num = np.linalg.norm(X_true - X_rec, ord='fro')
    denom = np.linalg.norm(X_true, ord='fro')
    return (num / denom) * 100


err = reconstruction_error(X_V, X_rec)
print(f"✅ VQPCA Relative Reconstruction Error: {err:.4f} %")
