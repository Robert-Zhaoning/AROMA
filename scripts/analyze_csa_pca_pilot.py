
import numpy as np

from sklearn.decomposition import PCA


PATH="outputs/csa_gradient_collection_v2/gradient_matrix.npy"


G=np.load(PATH)


print("="*70)
print("AROMA 2.0 CSA PCA PILOT")
print("="*70)


print(
    "Gradient matrix:",
    G.shape
)


# normalize each gradient
G_norm = (
    G /
    (
        np.linalg.norm(G,axis=1,keepdims=True)
        +1e-12
    )
)


pca=PCA()

pca.fit(G_norm)


explained=pca.explained_variance_ratio_


print("\nTop components:")

for k in [1,2,4,8,16,32]:

    if k<=len(explained):

        print(
            f"E{k}:",
            explained[:k].sum()
        )


print("\nEigen spectrum:")

print(
    explained[:10]
)


print("="*70)

