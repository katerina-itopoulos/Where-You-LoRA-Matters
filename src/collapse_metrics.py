import torch, json
import torch.nn.functional as F

EPS = 1e-8

def _to_cpu_f32(x):
    return None if x is None else x.detach().to(torch.float32, copy=False).cpu()

def _l2norm(X):  # [N,D]
    return X / (X.norm(dim=-1, keepdim=True).clamp_min(EPS))

def intra_modal_similarity(X):
    Xn = _l2norm(X); S = Xn @ Xn.T
    n = S.size(0)
    if n <= 1: return float("nan")
    mask = ~torch.eye(n, dtype=torch.bool)
    return S[mask].mean().item()

def intermodal_metrics(I, T):
    In, Tn = _l2norm(I), _l2norm(T)
    pos = (In * Tn).sum(-1)                            # matched pairs
    perm = torch.randperm(In.size(0))
    neg = (In * Tn[perm]).sum(-1)                      # mismatched
    margin = (pos.mean() - neg.mean()).item()
    auc = (pos[:,None] > neg[None,:]).float().mean().item() if pos.numel() and neg.numel() else float("nan")
    return {"pos_mean":pos.mean().item(), "neg_mean":neg.mean().item(),
            "margin":margin, "auc_proxy":auc}

def modality_gap(I, T, center=True):
    # I, T: [N, D] pooled image and text embeddings
    I = F.normalize(I, p=2, dim=-1)
    T = F.normalize(T, p=2, dim=-1)
    if center:
        I = I - I.mean(0, keepdim=True)
        T = T - T.mean(0, keepdim=True)
    mu_i = I.mean(0, keepdim=True)   # [1, D]
    mu_t = T.mean(0, keepdim=True)   # [1, D]
    # cosine distance between modality means (in [0, 2])
    gap = 1.0 - F.cosine_similarity(mu_i, mu_t).item()
    return gap

def effective_rank(Z, center=True):
    Z = Z.float()
    if center: Z = Z - Z.mean(0, keepdim=True)
    S = torch.linalg.svd(Z.to(torch.float64), full_matrices=False).S
    if S.numel() == 0: return float("nan")
    p = (S / S.sum().clamp_min(EPS)).clamp_min(EPS)
    H = -(p * p.log()).sum()
    return torch.exp(H).item()

def concentration_ratio(Z, k=1, center=True):
    Z = Z.float()
    if center: Z = Z - Z.mean(0, keepdim=True)
    S = torch.linalg.svd(Z.to(torch.float64), full_matrices=False).S
    if S.numel() == 0: return float("nan")
    k = min(k, S.numel())
    return (S[:k].sum() / S.sum().clamp_min(EPS)).item()

def linear_cka(X, Y, center=True):
    X = X.float(); Y = Y.float()
    if center:
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.T; L = Y @ Y.T
    num = (K * L).sum()
    den = torch.sqrt((K*K).sum()) * torch.sqrt((L*L).sum())
    return (num / den).item() if den > EPS else float("nan")

def summarize_vectors(V0_all, Vproj_all, T_all):
    V0 = _to_cpu_f32(V0_all)
    V  = _to_cpu_f32(Vproj_all)
    T  = _to_cpu_f32(T_all)

    assert V is not None and T is not None, "Need Vproj_all and T_all"
    assert V.dim()==2 and T.dim()==2, f"Expect [N,D], got {tuple(V.shape)} & {tuple(T.shape)}"
    assert V.shape[0]==T.shape[0], "V/T batch sizes differ; ensure you pooled per-sample vectors"

    metrics = {
        "intra_sim_img": intra_modal_similarity(V),
        "intra_sim_txt": intra_modal_similarity(T),
        "modality_gap":  modality_gap(V, T),
        **intermodal_metrics(V, T),
        "erank_img":     effective_rank(V),
        "erank_txt":     effective_rank(T),
        "cr1_img":       concentration_ratio(V, k=1),
        "cr1_txt":       concentration_ratio(T, k=1),
        "cka_img_txt":   linear_cka(V, T),
    }
    if V0 is not None:
        metrics.update({
            "cka_V0_Vproj": linear_cka(V0, V),
            "intra_sim_V0": intra_modal_similarity(V0),
            "erank_V0":     effective_rank(V0),
            "cr1_V0":       concentration_ratio(V0, k=1),
        })
    return metrics