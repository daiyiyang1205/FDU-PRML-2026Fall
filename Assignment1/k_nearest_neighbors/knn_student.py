import os
import numpy as np
from collections import Counter
from typing import List, Tuple
from data_generate import load_prepared_dataset
from viz_knn import plot_k_curve, plot_decision_boundary_multi

# 输出目录
OUT_DIR = "./output"
DATA_DIR = "./input_knn"

# ============ TODO 1：pairwise_dist ============
def pairwise_dist(X_test, X_train, metric, mode):
    """
    Compute pairwise distances between X_test (Nte,D) and X_train (Ntr,D).

    Required:
      - L2 distance 'l2' with modes:
          * 'two_loops'  
          * 'no_loops' 
      - 'cosine' distance (distance = 1 - cosine_similarity)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    Nte, D  = X_test.shape
    Ntr, D2 = X_train.shape
    assert D == D2, "Dim mismatch between test and train."

    if metric == "l2":
        if mode == "two_loops":
            # =============== TODO (students, REQUIRED) ===============
            # 双重循环：逐对计算欧式距离
            dists = np.zeros((Nte, Ntr), dtype=np.float64)
            for i in range(Nte):
                for j in range(Ntr):
                    diff = X_test[i] - X_train[j]
                    dists[i, j] = np.sqrt(np.sum(diff * diff))
            return dists
            # =========================================================

        elif mode == "no_loops":
            # =============== TODO (students, REQUIRED) ===============
            # 展开平方距离：||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i·x_j
            test_sq = np.sum(X_test ** 2, axis=1)       # (Nte,)
            train_sq = np.sum(X_train ** 2, axis=1)     # (Ntr,)
            cross = X_test @ X_train.T                  # (Nte, Ntr)
            sq_dists = test_sq[:, None] + train_sq[None, :] - 2.0 * cross
            # 数值误差可能使平方距离出现接近 0 的负值，先截断再开方
            sq_dists = np.maximum(sq_dists, 0.0)
            return np.sqrt(sq_dists)
            # =========================================================

        else:
            raise ValueError("Unknown mode for L2.")

    elif metric == "cosine":
        # =============== TODO (students, REQUIRED) ===============
        # 余弦距离 = 1 - 余弦相似度；分母过小时用 eps 兜底
        eps = 1e-12
        norm_te = np.linalg.norm(X_test, axis=1)    # (Nte,)
        norm_tr = np.linalg.norm(X_train, axis=1)   # (Ntr,)
        denom = np.maximum(norm_te[:, None] * norm_tr[None, :], eps)
        cosine_sim = (X_test @ X_train.T) / denom
        return 1.0 - cosine_sim
        # ================================================
    else:
        raise ValueError("metric must be 'l2' or 'cosine'.")


# ============ TODO 2：knn_predict（多数表决） ============
def knn_predict(X_test, X_train, y_train, k, metric, mode):
    """
    kNN prediction.
    Required: majority vote with L2 distance.

    Returns
    -------
    y_pred : (Nte,) int
    """
    dists = pairwise_dist(X_test, X_train, metric=metric, mode=mode)
    y_train = np.asarray(y_train).reshape(-1).astype(int)
    Nte = dists.shape[0]
    y_pred = np.zeros(Nte, dtype=int)

    for i in range(Nte):
        idx = np.argsort(dists[i], kind="stable")[:k]
        neighbors = y_train[idx]

        # =============== TODO (students, REQUIRED) ===============
        # 多数表决：统计 k 个近邻中各类别出现次数
        counter = Counter(neighbors.tolist())
        max_cnt = max(counter.values())
        # 平票时返回最小标签
        y_pred[i] = min(lab for lab, cnt in counter.items() if cnt == max_cnt)
        # ===========================================

    return y_pred


# ============ TODO 3：select_k_by_validation ============
def select_k_by_validation(X_train, y_train, X_val, y_val, ks: List[int], metric, mode) -> Tuple[int, List[float]]:
    """
    Grid-search K on validation set.

    Returns
    -------
    best_k : int
    accs   : list of validation accuracies aligned with ks
    """
    # =============== TODO (students, REQUIRED) ===============
    # 在 ks 上网格搜索：每个 k 在验证集上计算准确率
    y_val = np.asarray(y_val).reshape(-1)
    accs: List[float] = []
    for k in ks:
        y_pred = knn_predict(X_val, X_train, y_train, k,
                             metric=metric, mode=mode)
        accs.append(float(np.mean(y_pred == y_val)))

    # 取验证准确率最高的 k；平票时返回最小的 k
    max_acc = max(accs)
    best_k = min(k for k, a in zip(ks, accs) if a == max_acc)
    return best_k, accs
    # =========================================================


def run_with_visualization(metric="l2", mode="no_loops"):
    X_train, y_train, X_val, y_val, X_test, y_test = load_prepared_dataset(DATA_DIR)

    ks = [1, 3, 5, 7, 9, 11, 13]

    best_k, accs = select_k_by_validation(X_train, y_train, X_val, y_val,
                                          ks, metric=metric, mode=mode)
    print(f"[ModelSelect] metric={metric} best k={best_k} (val acc={max(accs):.4f})")
    plot_k_curve(ks, accs, os.path.join(OUT_DIR, f"knn_k_curve_{metric}.png"))

    X_trv = np.vstack([X_train, X_val]); y_trv = np.hstack([y_train, y_val])
    def predict_fn_for_k(k):
        return lambda Xq: knn_predict(Xq, X_trv, y_trv, k, metric=metric, mode=mode)

    ks_panel = sorted(set(ks + [best_k]))
    plot_decision_boundary_multi(predict_fn_for_k, X_train, y_train, X_test, y_test,
                                 ks=ks_panel,
                                 out_path=os.path.join(OUT_DIR, f"knn_boundary_grid_{metric}.png"),
                                 grid_n=200, batch_size=4096)


if __name__ == "__main__":
    # metric: ["l2", "cosine"];  mode: ["two_loops", "no_loops"]
    for metric in ["l2", "cosine"]:
        run_with_visualization(metric=metric, mode="no_loops")
