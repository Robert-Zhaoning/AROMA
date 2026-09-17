
import numpy as np
from sklearn.decomposition import PCA


INPUT="outputs/csa_gradient_v3/gradients.npy"
OUTPUT="outputs/csa_gradient_v3/csa_basis.npy"


G=np.load(INPUT)


print("Gradient matrix:",G.shape)


G=G-G.mean(axis=0,keepdims=True)


pca=PCA(
    n_components=50
)

pca.fit(G)


basis=pca.components_


print(
    "Basis shape:",
    basis.shape
)


np.save(
    OUTPUT,
    basis
)


print(
    "Saved:",
    OUTPUT
)


for k in [1,5,10,20]:

    print(
        "k=",
        k,
        "variance=",
        pca.explained_variance_ratio_[:k].sum()
    )

