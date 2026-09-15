
import numpy as np
from sklearn.decomposition import PCA


print("="*80)
print("AROMA CSA PCA ANALYSIS v1")
print("="*80)


PATH="outputs/csa_gradient_v3/gradients.npy"


G=np.load(PATH)


print(
    "GRADIENT MATRIX:",
    G.shape
)


# remove mean

G=G-G.mean(axis=0,keepdims=True)


# PCA

pca=PCA()

pca.fit(G)


var=pca.explained_variance_ratio_


print("\nEXPLAINED VARIANCE")


for k in [1,2,5,10,20,50,100]:

    print(
        f"Top {k}:",
        var[:k].sum()
    )


print("\nFIRST COMPONENTS")

for i,x in enumerate(var[:20]):

    print(
        f"PC{i+1}: {x}"
    )


# random null

print("\nRANDOM NULL")


R=np.random.randn(
    *G.shape
)

R=R-R.mean(
    axis=0,
    keepdims=True
)


rpca=PCA()

rpca.fit(R)


rvar=rpca.explained_variance_ratio_


for k in [1,2,5,10,20,50]:

    print(
        f"Random Top {k}:",
        rvar[:k].sum()
    )


print("="*80)
print("DONE")

