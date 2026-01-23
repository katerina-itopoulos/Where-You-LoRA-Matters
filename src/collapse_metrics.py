import torch
import torch.nn.functional as F

EPS = 1e-8

def _to_cpu_f32(x):
    return x.detach().to(torch.float32, copy=False).cpu()

def _l2norm(X):  # [N,D]
    return X / (X.norm(dim=-1, keepdim=True).clamp_min(EPS))

# -----------------------------
# Intra-modal similarity
# -----------------------------

def intra_modal_similarity(X, return_std=True):
    Xn = _l2norm(X)
    S = Xn @ Xn.T
    n = S.size(0)
    if n <= 1:
        return {"mean": float("nan"), "std": float("nan")}
    mask = ~torch.eye(n, dtype=torch.bool)
    vals = S[mask]
    return {
        "mean": vals.mean().item(),
        "std":  vals.std().item() if return_std else None
    }

# -----------------------------
# Inter-modal metrics
# -----------------------------

def intermodal_metrics(I, T):
    In, Tn = _l2norm(I), _l2norm(T)
    pos = (In * Tn).sum(-1)                      # matched pairs
    perm = torch.randperm(In.size(0))
    neg = (In * Tn[perm]).sum(-1)                # mismatched

    margin_vals = pos - neg

    auc = (pos[:, None] > neg[None, :]).float().mean().item() \
        if pos.numel() and neg.numel() else float("nan")

    return {
        "pos_mean": pos.mean().item(),
        "pos_std":  pos.std().item(),
        "neg_mean": neg.mean().item(),
        "neg_std":  neg.std().item(),
        "margin":   margin_vals.mean().item(),
        "margin_std": margin_vals.std().item(),
        "auc_proxy": auc
    }

# -----------------------------
# Modality gap
# -----------------------------

def modality_gap(I, T, center=True, return_std=True):
    I = F.normalize(I, p=2, dim=-1)
    T = F.normalize(T, p=2, dim=-1)
    if center:
        I = I - I.mean(0, keepdim=True)
        T = T - T.mean(0, keepdim=True)

    # per-sample gap (distance to other modality mean)
    mu_i = I.mean(0, keepdim=True)
    mu_t = T.mean(0, keepdim=True)

    gap_i = 1.0 - F.cosine_similarity(I, mu_t)
    gap_t = 1.0 - F.cosine_similarity(T, mu_i)
    gap_vals = torch.cat([gap_i, gap_t], dim=0)

    return {
        "mean": gap_vals.mean().item(),
        "std":  gap_vals.std().item() if return_std else None
    }

# -----------------------------
# Global geometry metrics
# -----------------------------

def effective_rank(Z, center=True):
    Z = Z.float()
    if center:
        Z = Z - Z.mean(0, keepdim=True)
    S = torch.linalg.svd(Z.to(torch.float64), full_matrices=False).S
    if S.numel() == 0:
        return float("nan")
    p = (S / S.sum().clamp_min(EPS)).clamp_min(EPS)
    H = -(p * p.log()).sum()
    return torch.exp(H).item()

def concentration_ratio(Z, k=1, center=True):
    Z = Z.float()
    if center:
        Z = Z - Z.mean(0, keepdim=True)
    S = torch.linalg.svd(Z.to(torch.float64), full_matrices=False).S
    if S.numel() == 0:
        return float("nan")
    k = min(k, S.numel())
    return (S[:k].sum() / S.sum().clamp_min(EPS)).item()

def linear_cka(X, Y, center=True):
    X = X.float()
    Y = Y.float()
    if center:
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.T
    L = Y @ Y.T
    num = (K * L).sum()
    den = torch.sqrt((K * K).sum()) * torch.sqrt((L * L).sum())
    return (num / den).item() if den > EPS else float("nan")

# -----------------------------
# Main API
# -----------------------------

def summarize_vectors(Vproj_all, T_all):
    V = _to_cpu_f32(Vproj_all)
    T = _to_cpu_f32(T_all)

    assert V is not None and T is not None
    assert V.dim() == 2 and T.dim() == 2
    assert V.shape[0] == T.shape[0]

    intra_img = intra_modal_similarity(V)
    intra_txt = intra_modal_similarity(T)
    gap = modality_gap(V, T)
    inter = intermodal_metrics(V, T)

    metrics = {
        # Intra
        "intra_sim_img_mean": intra_img["mean"],
        "intra_sim_img_std":  intra_img["std"],
        "intra_sim_txt_mean": intra_txt["mean"],
        "intra_sim_txt_std":  intra_txt["std"],

        # Inter
        **inter,

        # Modality gap
        "modality_gap_mean": gap["mean"],
        "modality_gap_std":  gap["std"],

        # Global geometry
        "erank_img":   effective_rank(V),
        "erank_txt":   effective_rank(T),
        "cr1_img":     concentration_ratio(V, k=1),
        "cr1_txt":     concentration_ratio(T, k=1),
        "cka_img_txt": linear_cka(V, T),
    }

    return metrics
