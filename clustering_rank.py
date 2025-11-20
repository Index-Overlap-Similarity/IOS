import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from scipy.optimize import linear_sum_assignment


# --- helper functions (from previous code) ---
def row_normalize_counts(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    rs = X.sum(axis=1, keepdims=True) + eps
    return X / rs


def hellinger_embed(P):
    return np.sqrt(np.clip(P, 0.0, 1.0))


def js_distance_matrix(P, eps=1e-12, base=2.0):
    P = np.clip(P, eps, 1.0)
    N = P.shape[0]
    D = np.zeros((N, N), dtype=float)
    logP = np.log(P) / np.log(base)
    for i in range(N):
        for j in range(i + 1, N):
            M = 0.5 * (P[i] + P[j])
            js = 0.5 * (
                (P[i] * (logP[i] - (np.log(M) / np.log(base)))).sum()
                + (P[j] * (logP[j] - (np.log(M) / np.log(base)))).sum()
            )
            D_ij = np.sqrt(max(js, 0.0))
            D[i, j] = D[j, i] = D_ij
    return D


def clustering_indices_from_counts(X_counts, y_clusters):
    P = row_normalize_counts(X_counts)
    Phi = hellinger_embed(P)

    # Silhouette with JS
    D_js = js_distance_matrix(P)
    sil_js = silhouette_score(D_js, y_clusters, metric="precomputed")

    # Silhouette with Hellinger (Euclidean on sqrt probabilities)
    norms = (Phi**2).sum(axis=1, keepdims=True)
    D2 = norms + norms.T - 2 * (Phi @ Phi.T)
    D2 = np.maximum(D2, 0.0)
    D_hell = np.sqrt(D2)
    sil_hell = silhouette_score(D_hell, y_clusters, metric="precomputed")

    db = davies_bouldin_score(Phi, y_clusters)
    ch = calinski_harabasz_score(Phi, y_clusters)

    return {
        "silhouette_JS": sil_js,
        "silhouette_Hellinger": sil_hell,
        "DB_Hellinger": db,
        "CH_Hellinger": ch,
    }


# --- clustering-accuracy helpers ---
def purity_score(y_true, y_pred):
    # compute contingency matrix then Hungarian matching
    labels_true = np.asarray(y_true)
    labels_pred = np.asarray(y_pred)
    true_labels = np.unique(labels_true)
    pred_labels = np.unique(labels_pred)
    # build contingency
    cont = np.zeros((true_labels.size, pred_labels.size), dtype=int)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            cont[i, j] = np.sum((labels_true == t) & (labels_pred == p))
    # maximize sum over matching
    row_ind, col_ind = linear_sum_assignment(-cont)  # negate to maximize
    return cont[row_ind, col_ind].sum() / labels_true.size


# --- your input data (clients x 10 classes) ---
X_counts = np.array(
    [
        [12, 0, 2023, 0, 53, 2865, 4509, 0, 0, 0],  # client_0, FL=6
        [0, 0, 98, 288, 7, 38, 825, 9, 1690, 0],  # client_1, FL=12
        [3, 4978, 3691, 0, 0, 0, 0, 0, 0, 0],  # client_2, FL=18
        [4641, 246, 0, 1, 0, 0, 0, 3, 0, 23],  # client_3, FL=24
        [830, 2, 110, 4, 0, 0, 590, 1666, 15, 5976],  # client_4, FL=30
        [172, 0, 74, 0, 0, 0, 75, 0, 780, 0],  # client_5, FL=36
        [0, 685, 0, 5706, 0, 0, 0, 0, 0, 0],  # client_6, FL=42
        [341, 19, 2, 0, 0, 3096, 0, 1, 3465, 0],  # client_7, FL=48
        [0, 69, 1, 0, 5938, 0, 0, 0, 0, 0],  # client_8, FL=54
        [1, 1, 1, 1, 2, 1, 1, 4321, 50, 1],  # client_9, FL=60
    ],
    dtype=float,
)

fl_rounds_per_client = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60], dtype=int)

# Data-Driven ground truth grouping (global groups)
groups_data_driven = {
    0: [0, 1, 2, 5, 6, 7],  # cluster 0 (class indices)
    1: [3, 4, 9],  # cluster 1
    2: [8],  # cluster 2
}

# Per-client mappings as in your table
coordinate_mappings = [
    np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 2]),  # client_0
    np.array([0, 0, 0, 0, 1, 0, 2, 1, 0, 1]),  # client_1
    np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),  # client_2
    np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),  # client_3
    np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),  # client_4
    np.array([0, 0, 0, 0, 2, 0, 1, 2, 0, 2]),  # client_5
    np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),  # client_6
    np.array([0, 0, 0, 0, 2, 0, 1, 2, 0, 2]),  # client_7
    np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),  # client_8
    np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),  # client_9
]

common_map = np.array([0, 2, 2, 2, 1, 2, 2, 3, 2, 2])
cosine_grad_mappings = [common_map.copy() for _ in range(X_counts.shape[0])]
cosine_param_mappings = [common_map.copy() for _ in range(X_counts.shape[0])]
euclidean_mappings = [common_map.copy() for _ in range(X_counts.shape[0])]


# --- assignment helpers (re-used) ---
def assign_clusters_from_global_groups(X, groups):
    N = X.shape[0]
    y = np.zeros(N, dtype=int)
    keys = sorted(groups.keys())
    for i in range(N):
        sums = [X[i, groups[k]].sum() for k in keys]
        y[i] = keys[int(np.argmax(sums))]
    return y


def assign_clusters_from_per_client_mappings(X, mappings):
    N, L = X.shape
    y = np.zeros(N, dtype=int)
    for i in range(N):
        map_i = np.asarray(mappings[i], dtype=int)
        labels = np.unique(map_i)
        sums = [X[i, map_i == lab].sum() for lab in labels]
        y[i] = int(labels[int(np.argmax(sums))])
    return y


# --- evaluate at each round threshold ---
round_thresholds = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

methods_mappings = {
    "Coordinate": coordinate_mappings,
    "Cosine Param": cosine_param_mappings,
    "Cosine Grads": cosine_grad_mappings,
    "Euclidean": euclidean_mappings,
}

# header for printing
print(
    "Round | Method        |  N_clients | purity   |   ARI    |   NMI    | sil_JS  | sil_Hel | DB_Hel  | CH_Hel"
)
print("-" * 110)

for r in round_thresholds:
    # select clients up to round r
    idx = np.where(fl_rounds_per_client <= r)[0]
    if idx.size == 0:
        continue
    X_sub = X_counts[idx]
    # ground truth labels on subset
    y_true_sub = assign_clusters_from_global_groups(X_sub, groups_data_driven)

    for method_name, mappings in methods_mappings.items():
        # filter mappings to subset
        maps_sub = [mappings[i] for i in idx]
        y_pred_sub = assign_clusters_from_per_client_mappings(X_sub, maps_sub)

        # external clustering accuracy metrics
        purity = purity_score(y_true_sub, y_pred_sub)
        ari = adjusted_rand_score(y_true_sub, y_pred_sub)
        nmi = normalized_mutual_info_score(y_true_sub, y_pred_sub)

        # internal clustering indices: require at least 2 clusters
        if len(np.unique(y_pred_sub)) >= 2:
            try:
                internal = clustering_indices_from_counts(X_sub, y_pred_sub)
                sil_js = internal["silhouette_JS"]
                sil_hell = internal["silhouette_Hellinger"]
                db = internal["DB_Hellinger"]
                ch = internal["CH_Hellinger"]
            except Exception:
                sil_js = sil_hell = db = ch = float("nan")
        else:
            sil_js = sil_hell = db = ch = float("nan")

        print(
            f"{r:5d} | {method_name:13s} | {idx.size:10d} | {purity:7.4f} | {ari:7.4f} | {nmi:7.4f} |"
            f" {np.nan if sil_js!=sil_js else sil_js:7.4f} | {np.nan if sil_hell!=sil_hell else sil_hell:7.4f} |"
            f" {np.nan if db!=db else db:7.4f} | {np.nan if ch!=ch else ch:7.4f}"
        )

    # also evaluate Data-Driven grouping itself as a trivial baseline (predict = ground-truth)
    y_dd_sub = assign_clusters_from_global_groups(X_sub, groups_data_driven)
    purity = purity_score(y_true_sub, y_dd_sub)
    ari = adjusted_rand_score(y_true_sub, y_dd_sub)
    nmi = normalized_mutual_info_score(y_true_sub, y_dd_sub)
    # silhouette etc. for ground-truth labels
    if len(np.unique(y_dd_sub)) >= 2:
        internal = clustering_indices_from_counts(X_sub, y_dd_sub)
        sil_js = internal["silhouette_JS"]
        sil_hell = internal["silhouette_Hellinger"]
        db = internal["DB_Hellinger"]
        ch = internal["CH_Hellinger"]
    else:
        sil_js = sil_hell = db = ch = float("nan")
    print(
        f"{r:5d} | {'Data-Driven':13s} | {idx.size:10d} | {purity:7.4f} | {ari:7.4f} | {nmi:7.4f} |"
        f" {np.nan if sil_js!=sil_js else sil_js:7.4f} | {np.nan if sil_hell!=sil_hell else sil_hell:7.4f} |"
        f" {np.nan if db!=db else db:7.4f} | {np.nan if ch!=ch else ch:7.4f}"
    )

    print("-" * 110)
