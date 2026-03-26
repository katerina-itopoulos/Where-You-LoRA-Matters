import torch
import torch.nn.functional as F

EPS = 1e-8


def _to_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32, copy=False).cpu()


def _l2norm(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=-1, keepdim=True).clamp_min(EPS)


def _safe_svd(Z: torch.Tensor) -> torch.Tensor:
    """SVD with Frobenius normalisation and jitter for numerical stability."""
    Z64 = Z.to(torch.float64)
    frob = Z64.norm()
    if frob > 1e-8:
        Z64 = Z64 / frob
    Z64 = Z64 + torch.randn_like(Z64) * 1e-6
    try:
        return torch.linalg.svd(Z64, full_matrices=False).S
    except torch.linalg.LinAlgError:
        return torch.linalg.svdvals(Z64)


def intra_modal_similarity(X: torch.Tensor, return_std: bool = True) -> dict[str, float]:
    """Return mean and std of pairwise cosine similarities within a modality."""
    Xn = _l2norm(X)
    S = Xn @ Xn.T
    n = S.size(0)
    if n <= 1:
        return {"mean": float("nan"), "std": float("nan")}
    vals = S[~torch.eye(n, dtype=torch.bool)]
    return {"mean": vals.mean().item(), "std": vals.std().item() if return_std else None}


def intermodal_metrics(I: torch.Tensor, T: torch.Tensor) -> dict[str, float]:
    """Return matched-pair similarity, negative similarity, margin, and AUC proxy."""
    In, Tn = _l2norm(I), _l2norm(T)
    n = In.size(0)
    pos = (In * Tn).sum(-1)

    neg1 = (In * Tn[torch.roll(torch.arange(n), shifts=1)]).sum(-1)
    neg2 = (In * Tn[torch.roll(torch.arange(n), shifts=-1)]).sum(-1)
    neg = torch.stack([neg1, neg2], -1).mean(-1)

    margin_vals = pos - neg
    auc = (
        (pos[:, None] > neg[None, :]).float().mean().item()
        if pos.numel() and neg.numel()
        else float("nan")
    )

    return {
        "pos_mean": pos.mean().item(),
        "pos_std": pos.std().item(),
        "neg_mean": neg.mean().item(),
        "neg_std": neg.std().item(),
        "margin": margin_vals.mean().item(),
        "margin_std": margin_vals.std().item(),
        "auc_proxy": auc,
    }


def modality_gap(
    I: torch.Tensor,
    T: torch.Tensor,
    center: bool = True,
    return_std: bool = True,
) -> dict[str, float]:
    """Return mean and std of the cosine distance between each sample and the opposite-modality centroid."""
    I = F.normalize(I, p=2, dim=-1)
    T = F.normalize(T, p=2, dim=-1)

    if center:
        I = I - I.mean(0, keepdim=True)
        T = T - T.mean(0, keepdim=True)

    mu_i = I.mean(0, keepdim=True)
    mu_t = T.mean(0, keepdim=True)

    gap_vals = torch.cat([1.0 - F.cosine_similarity(I, mu_t), 1.0 - F.cosine_similarity(T, mu_i)])
    return {"mean": gap_vals.mean().item(), "std": gap_vals.std().item() if return_std else None}


def effective_rank(Z: torch.Tensor, center: bool = True) -> dict[str, float]:
    """Return effective rank and normalised effective rank via singular value entropy."""
    Z = Z.float()
    if center:
        Z = Z - Z.mean(0, keepdim=True)

    print(f"Z shape: {Z.shape}, std: {Z.std().item():.6f}, max: {Z.abs().max().item():.6f}")

    S = _safe_svd(Z)
    if S.numel() == 0:
        return {"erank": float("nan"), "norm_erank": float("nan")}

    p = (S / S.sum().clamp_min(EPS)).clamp_min(EPS)
    erank = torch.exp(-(p * p.log()).sum()).item()
    return {"erank": erank, "norm_erank": erank / min(Z.shape[0], Z.shape[1])}


def concentration_ratio(Z: torch.Tensor, k: int = 1, center: bool = True) -> float:
    """Return the fraction of total singular value mass captured by the top-k components."""
    Z = Z.float()
    if center:
        Z = Z - Z.mean(0, keepdim=True)
    S = _safe_svd(Z)
    if S.numel() == 0:
        return float("nan")
    return (S[: min(k, S.numel())].sum() / S.sum().clamp_min(EPS)).item()


def centered_kernel_alignment(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Return linear CKA between X and Y using double-centred HSIC."""
    X, Y = X.float(), Y.float()
    n = X.size(0)
    if n <= 1:
        return float("nan")

    K = X @ X.T
    L = Y @ Y.T
    H = torch.eye(n, device=K.device) - torch.ones((n, n), device=K.device) / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    denom = (n - 1) ** 2
    hsic_kl = (Kc * Lc).sum() / denom
    hsic_kk = (Kc * Kc).sum() / denom
    hsic_ll = (Lc * Lc).sum() / denom

    return (hsic_kl / (hsic_kk * hsic_ll).sqrt()).clamp(0, 1).item()


def summarize_vectors(Vproj_all: torch.Tensor, T_all: torch.Tensor) -> dict[str, float]:
    """Compute the full suite of modality collapse metrics for projected visual and text vectors."""
    V = _to_cpu_f32(Vproj_all)
    T = _to_cpu_f32(T_all)

    assert V.dim() == 2 and T.dim() == 2
    assert V.shape[0] == T.shape[0]

    intra_img = intra_modal_similarity(V)
    intra_txt = intra_modal_similarity(T)
    gap = modality_gap(V, T)
    inter = intermodal_metrics(V, T)
    erank_img = effective_rank(V)
    erank_txt = effective_rank(T)

    return {
        "intra_sim_img_mean": intra_img["mean"],
        "intra_sim_img_std": intra_img["std"],
        "intra_sim_txt_mean": intra_txt["mean"],
        "intra_sim_txt_std": intra_txt["std"],
        **inter,
        "modality_gap_mean": gap["mean"],
        "modality_gap_std": gap["std"],
        "erank_img": erank_img["erank"],
        "norm_erank_img": erank_img["norm_erank"],
        "erank_txt": erank_txt["erank"],
        "norm_erank_txt": erank_txt["norm_erank"],
        "cr1_img": concentration_ratio(V, k=1),
        "cr1_txt": concentration_ratio(T, k=1),
        "cka_img_txt": centered_kernel_alignment(V, T),
    }