from __future__ import annotations
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.cluster import KMeans


def tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        npdatamax = xmax.view(n_heads, -1, n).cpu().numpy()
        npdatamin = xmin.view(n_heads, -1, n).cpu().numpy()
    else:
        npdatamax = xmax.view(n_heads, -1, 1).cpu().numpy()
        npdatamin = xmin.view(n_heads, -1, 1).cpu().numpy()
    npdata = np.concatenate([npdatamax, npdatamin], -1)
    cnt = 0
    all_index = []
    all_counts = []
    for data in npdata:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(data)
        counts = np.bincount(kmeans.labels_)
        labels = torch.from_numpy(kmeans.labels_)
        index = torch.argsort(labels)
        index += cnt
        all_index.append(index)
        all_counts.append(counts)
        cnt += len(data)
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def peg_tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1).cpu()
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1).cpu()
        tdatamax = xmax.view(n_heads, -1, n)
        tdatamin = xmin.view(n_heads, -1, n)
        tdata = (tdatamax[:, :, 0] - tdatamin[:, :, 0]).reshape(n_heads, -1)
    else:
        tdatamax = xmax.view(n_heads, -1, 1).cpu()
        tdatamin = xmin.view(n_heads, -1, 1).cpu()
        tdata = (tdatamax - tdatamin).reshape(n_heads, -1)
    cnt = 0
    all_index = []
    all_counts = []
    for data in tdata:
        index = torch.argsort(data)
        counts = [int(index.numel()) // n_clusters] * n_clusters
        index = index + cnt
        all_index.append(index)
        all_counts.append(np.array(counts))
        cnt += index.numel()
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def interval_tensor_calc_reorder_index(xmax, xmin, n_clusters, n_heads=None, key: str = 'center'):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1).cpu()
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1).cpu()
        tmax = xmax.view(n_heads, -1, n)[:, :, 0]
        tmin = xmin.view(n_heads, -1, n)[:, :, 0]
    else:
        tmax = xmax.view(n_heads, -1).cpu()
        tmin = xmin.view(n_heads, -1).cpu()
    center = (tmin + tmax) / 2
    width = (tmax - tmin)
    cnt = 0
    all_index = []
    all_counts = []
    for c_row, w_row, xmin_row, xmax_row in zip(center, width, tmin, tmax):
        if key == 'center':
            keys = torch.stack([c_row, w_row], dim=0)
            idx_w = torch.argsort(keys[1], stable=True)
            c_sorted = keys[0][idx_w]
            idx_c = torch.argsort(c_sorted, stable=True)
            index = idx_w[idx_c]
        else:
            keys = torch.stack([xmin_row, xmax_row], dim=0)
            idx_xmax = torch.argsort(keys[1], stable=True)
            xmin_sorted = keys[0][idx_xmax]
            idx_xmin = torch.argsort(xmin_sorted, stable=True)
            index = idx_xmax[idx_xmin]
        counts = [int(index.numel()) // n_clusters] * n_clusters
        index = index + cnt
        all_index.append(index)
        all_counts.append(np.array(counts))
        cnt += index.numel()
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def balanced_kmeans2d_calc_reorder_index(xmax, xmin, n_clusters, block_size, n_heads=None):
    if n_heads is None:
        n_heads = 1
    if isinstance(xmax, list):
        n = len(xmax)
        xmax = torch.cat([_.unsqueeze(-1) for _ in xmax], -1)
        xmin = torch.cat([_.unsqueeze(-1) for _ in xmin], -1)
        np_xmax = xmax.view(n_heads, -1, n).cpu().numpy()
        np_xmin = xmin.view(n_heads, -1, n).cpu().numpy()
    else:
        np_xmax = xmax.view(n_heads, -1, 1).cpu().numpy()
        np_xmin = xmin.view(n_heads, -1, 1).cpu().numpy()
    cnt = 0
    all_index = []
    all_counts = []
    def _greedy_capacity(data_np: np.ndarray, K: int, cap_size: int):
        if data_np.ndim == 3:
            data_np = data_np[..., 0]
        D_head = data_np.shape[0]
        assert D_head == K * cap_size
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(data_np)
        centers = torch.from_numpy(km.cluster_centers_.astype(np.float32))
        pts = torch.from_numpy(data_np.astype(np.float32))
        dists = torch.cdist(pts, centers, p=2)
        prefs = torch.argsort(dists, dim=1).cpu().numpy()
        if K >= 2:
            d_sorted, _ = torch.sort(dists, dim=1)
            margin = (d_sorted[:, 1] - d_sorted[:, 0]).cpu().numpy()
        else:
            margin = np.full((D_head,), np.inf)
        order = np.argsort(-margin)
        cap = np.full((K,), cap_size, dtype=np.int32)
        assign = -np.ones((D_head,), dtype=np.int32)
        for i in order:
            for c in prefs[i]:
                if cap[c] > 0:
                    assign[i] = c
                    cap[c] -= 1
                    break
        assert (assign >= 0).all() and cap.sum() == 0
        return assign, centers, torch.cdist(pts, centers, p=2)
    for x_min, x_max in zip(np_xmin, np_xmax):
        data = np.concatenate([x_min, x_max], axis=-1)
        if data.ndim == 3:
            data = data[..., 0]
        D_head = data.shape[0]
        K = n_clusters
        assert D_head == K * block_size
        used_lib = False
        labels = None
        centers_t = None
        dists = None
        try:
            from kmeans_pytorch import KMeans as TorchKMeans
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            X = torch.from_numpy(data.astype(np.float32)).to(device).view(-1, 2)
            kmp = TorchKMeans(n_clusters=K, device=device, balanced=True)
            X = X.clamp(min=-1e6, max=1e6).nan_to_num(0.0)
            _ = kmp.fit(X=X, distance='euclidean', iter_limit=100, tqdm_flag=False)
            labels_t = kmp.predict(X=X)
            centers_dev = kmp.cluster_centers
            labels = labels_t.to('cpu', dtype=torch.long)
            centers_t = centers_dev.to('cpu', dtype=torch.float32)
            pts = torch.from_numpy(data.astype(np.float32))
            dists = torch.cdist(pts, centers_t, p=2)
            counts_chk = torch.bincount(labels, minlength=K).cpu().numpy()
            used_lib = np.all(counts_chk == block_size)
        except Exception:
            used_lib = False
        if not used_lib:
            assign_np, centers_t, dists = _greedy_capacity(data, K, block_size)
            labels = torch.from_numpy(assign_np)
        centers_center = centers_t.mean(dim=1)
        cluster_order = torch.argsort(centers_center).cpu().numpy().tolist()
        dnp = dists.cpu().numpy()
        idx_list = []
        counts = []
        for k in cluster_order:
            members = torch.nonzero(labels == k, as_tuple=False).view(-1).cpu().numpy()
            members = members[np.argsort(dnp[members, k])]
            idx_list.append(torch.from_numpy(members + cnt))
            counts.append(len(members))
        cnt += D_head
        all_index.append(torch.hstack(idx_list))
        all_counts.append(np.array(counts))
    all_index = torch.hstack(all_index)
    all_counts = np.hstack(all_counts)
    return all_index, all_counts


def hybrid_interval_kmeans_reorder_index(xmax: torch.Tensor, xmin: torch.Tensor, block_size: int, top_pct: float = 0.10, key: str = 'center') -> Tuple[torch.Tensor, np.ndarray]:
    xmx = xmax.view(-1).cpu()
    xmn = xmin.view(-1).cpu()
    D = xmx.numel()
    assert D == xmn.numel()
    assert D % block_size == 0
    K = D // block_size
    center = (xmn + xmx) / 2
    width = (xmx - xmn)
    if key == 'center':
        idx_w = torch.argsort(width, stable=True)
        c_sorted = center[idx_w]
        idx_c = torch.argsort(c_sorted, stable=True)
        idx_sorted = idx_w[idx_c]
    else:
        idx_xmax = torch.argsort(xmx, stable=True)
        xmin_sorted = xmn[idx_xmax]
        idx_xmin = torch.argsort(xmin_sorted, stable=True)
        idx_sorted = idx_xmax[idx_xmin]
    blocks = idx_sorted.view(K, block_size)
    top_pct = float(max(0.0, min(0.5, top_pct)))
    keep = int(round(K * top_pct))
    if keep == 0:
        idx_mid, counts_mid = balanced_kmeans2d_calc_reorder_index(xmax, xmin, K, block_size)
        return idx_mid.to(torch.long), counts_mid
    K_rem = K - 2 * keep
    if K_rem <= 0:
        return idx_sorted.to(torch.long), np.full((K,), block_size)
    left_keep = blocks[:keep].reshape(-1)
    right_keep = blocks[K - keep:].reshape(-1)
    mid_idx = blocks[keep:K - keep].reshape(-1)
    idx_mid_local, counts_mid = balanced_kmeans2d_calc_reorder_index(xmax[mid_idx], xmin[mid_idx], K_rem, block_size)
    mid_ordered = mid_idx[idx_mid_local]
    final_idx = torch.cat([left_keep, mid_ordered, right_keep], dim=0).to(torch.long)
    counts = np.hstack([np.full((keep,), block_size), counts_mid, np.full((keep,), block_size)])
    return final_idx, counts


def hybrid_plus_reorder_index(xmax: torch.Tensor, xmin: torch.Tensor, block_size: int, top_pct: float = 0.10) -> Tuple[torch.Tensor, np.ndarray]:
    xmx = xmax.view(-1).cpu()
    xmn = xmin.view(-1).cpu()
    D = xmx.numel()
    assert D == xmn.numel()
    assert D % block_size == 0
    K = D // block_size
    
    abs_max = torch.max(torch.abs(xmx), torch.abs(xmn))
    idx_abs_sorted = torch.argsort(abs_max, descending=True, stable=True)
    blocks = idx_abs_sorted.view(K, block_size)
    
    top_pct = float(max(0.0, min(0.5, top_pct)))
    keep = int(round(K * top_pct))
    
    if keep == 0:
        idx_mid, counts_mid = balanced_kmeans2d_calc_reorder_index(xmax, xmin, K, block_size)
        return idx_mid.to(torch.long), counts_mid
    
    K_rem = K - 2 * keep
    if K_rem <= 0:
        return idx_abs_sorted.to(torch.long), np.full((K,), block_size)
    
    left_keep = blocks[:keep].reshape(-1)
    right_keep = blocks[K - keep:].reshape(-1)
    mid_idx = blocks[keep:K - keep].reshape(-1)
    
    idx_mid_local, counts_mid = balanced_kmeans2d_calc_reorder_index(xmax[mid_idx], xmin[mid_idx], K_rem, block_size)
    mid_ordered = mid_idx[idx_mid_local]
    
    final_idx = torch.cat([left_keep, mid_ordered, right_keep], dim=0).to(torch.long)
    counts = np.hstack([np.full((keep,), block_size), counts_mid, np.full((keep,), block_size)])
    return final_idx, counts


@torch.no_grad()
def compute_reorders(oc_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]], block_size: int, method: str = 'peg', interval_key: str = 'center', acts_for_mse: Dict[str, torch.Tensor] | None = None, sls_iters: int = 20, sls_block_axis: str = 'feature', sls_scale_format: str = 'e4m3', hybrid_top_pct: float = 0.10) -> Dict[str, torch.Tensor]:
    perms: Dict[str, torch.Tensor] = {}
    for name, (xmax, xmin) in oc_stats.items():
        D = xmax.numel()
        if D % block_size != 0:
            continue
        n_clusters = D // block_size
        if method == 'kmeans':
            idx, counts = balanced_kmeans2d_calc_reorder_index(xmax, xmin, n_clusters, block_size)
        elif method == 'interval':
            idx, counts = interval_tensor_calc_reorder_index(xmax, xmin, n_clusters, key=interval_key)
        elif method == 'hybrid':
            idx, counts = hybrid_interval_kmeans_reorder_index(xmax, xmin, block_size, top_pct=hybrid_top_pct, key=interval_key)
        elif method == 'hybrid_plus':
            idx, counts = hybrid_plus_reorder_index(xmax, xmin, block_size, top_pct=hybrid_top_pct)
        else:
            idx, counts = peg_tensor_calc_reorder_index(xmax, xmin, n_clusters)
        if idx.numel() != D:
            continue
        perms[name] = idx.to(torch.long)
    return perms
