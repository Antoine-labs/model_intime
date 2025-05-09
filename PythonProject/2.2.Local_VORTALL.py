import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib
matplotlib.use('TkAgg')

# === 1. Load Data ===
data = scipy.io.loadmat('CYLINDER_ALL.mat')
UALL = data['UALL']
VALL = data['VALL']
VORTALL = data['VORTALL']
nx = data["ny"][0][0]
ny = data["nx"][0][0]
nt_U = UALL.shape[1]
nt_V = VALL.shape[1]
nt_VORT = VORTALL.shape[1]

X_U = UALL.T  # (nt, np)
X_V = VALL.T
X_VORT = VORTALL.T



# === 2. Standardize data ===
scaler = StandardScaler()
X0_U = scaler.fit_transform(X_U)
X0_V = scaler.fit_transform(X_V)
X0_VORT = scaler.fit_transform(X_VORT)

#########################################################################################
#%% Vortex

# ------------------------------------------------------------------------------
# 🧪 PARTIE 1 — Choix du nombre de clusters (via Davies-Bouldin Index)
# ------------------------------------------------------------------------------

# Range de k à tester
# Create the k_array and the db_score_array, initiliazed with zeros
# Définir la plage de k à tester
X_VORT_spatial = VORTALL  # (np, nt_U)
"""
# Range de k à tester
k_array = np.arange(2, 10)
db_score_array = np.zeros_like(k_array, dtype=float)

for i, k in enumerate(k_array):
    kmeans = KMeans(n_clusters=k, init='random', n_init='auto', random_state=0)
    kmeans.fit(X_VORT_spatial)  # chaque ligne = un point spatial
    labels = kmeans.labels_

    # DB-score sur les vecteurs temporels
    db_score = davies_bouldin_score(X_VORT_spatial, labels)
    db_score_array[i] = db_score

# Affichage du Davies-Bouldin score
plt.figure(figsize=(6, 4))
plt.plot(k_array, db_score_array, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Davies-Bouldin score')
plt.title('DB Score vs Number of Clusters')
plt.grid(True)
plt.tight_layout()
plt.show()

# Choix optimal
index = np.argmin(db_score_array)
n_clusters = k_array[index]
print(f'Optimal number of clusters: {n_clusters}')
"""
# ------------------------------------------------------------------------------
# 🔹 PARTIE 2 — Clustering des snapshots
# ------------------------------------------------------------------------------

optimal_k = 5  # choisis visuellement depuis le plot précédent
kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
cluster_labels = kmeans.fit_predict(X0_VORT)

# ------------------------------------------------------------------------------
# 🔁 PARTIE 3 — Local PCA dans chaque cluster + reconstruction
# ------------------------------------------------------------------------------

q = 6  # nombre de composantes principales à garder
X0_rec_local_VORT = np.zeros_like(X0_VORT)

for k in range(optimal_k):
    mask = (cluster_labels == k)
    X_k = X0_VORT[mask]

    scaler_k = StandardScaler()
    X_k_scaled = scaler_k.fit_transform(X_k)

    pca_k = PCA()
    pca_k.fit(X_k_scaled)
    Z_k = X_k_scaled @ pca_k.components_.T
    X0_k_rec = Z_k[:, :q] @ pca_k.components_[:q, :]

    X_k_rec = scaler_k.inverse_transform(X0_k_rec)
    X0_rec_local_VORT[mask] = X_k_rec

# Retour à l’espace original non standardisé
X_rec_local_VORT = scaler.inverse_transform(X0_rec_local_VORT)

# ------------------------------------------------------------------------------
# 📊 Visualisation d’une reconstruction
# ------------------------------------------------------------------------------

x = np.linspace(-1, 8, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y, indexing='ij')
t = 50  # timestep à visualiser

vort_orig = VORTALL[:, t].reshape((nx, ny))
vort_rec = X_rec_local_VORT[t, :].reshape((nx, ny))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
levels = np.linspace(-1.5, 1.5, 100)

cf1 = axes[0].contourf(X, Y, vort_orig, levels=levels, cmap='seismic')
axes[0].set_title(f'Original VORT (t={t})')
axes[0].set_xlabel('x / D')
axes[0].set_ylabel('y / D')

cf2 = axes[1].contourf(X, Y, vort_rec, levels=levels, cmap='seismic')
axes[1].set_title(f'Reconstructed VORT (Local PCA, q={q}, t={t})')
axes[1].set_xlabel('x / D')
axes[1].set_ylabel('y / D')

plt.colorbar(cf1, ax=axes, orientation='vertical', shrink=0.8)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------------------
# 🧮 (Bonus) Calcul de l’erreur globale
# ------------------------------------------------------------------------------

def reconstruction_error(X_original, X_reconstructed):
    num = np.linalg.norm(X_original - X_reconstructed, ord='fro')
    denom = np.linalg.norm(X_original, ord='fro')
    return (num / denom) * 100

err = reconstruction_error(X_VORT, X_rec_local_VORT)
print(f"Erreur relative de reconstruction (Local PCA): {err:.5f} %")