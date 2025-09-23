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
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0,max_iter=1000).fit(data)
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
        # 针对大量通道优化K-means参数
        if D_head > 1000:
            # 大量通道：使用K-means++初始化，增加迭代次数和初始化次数
            km = KMeans(n_clusters=K, n_init=20, random_state=0, init='k-means++', max_iter=500).fit(data_np)
        else:
            # 少量通道：使用默认参数
            km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=300).fit(data_np)
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
        
        # 数据预处理：标准化以提高聚类效果
        if D_head > 1000:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0) + 1e-8
            data_normalized = (data - data_mean) / data_std
            print(f"Applied data normalization for {D_head} channels")
        else:
            data_normalized = data
        # 直接使用sklearn的KMeans
        assign_np, centers_t, dists = _greedy_capacity(data_normalized, K, block_size)
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


def qk_proj_quadruple_reorder_index(q_xmax: torch.Tensor, q_xmin: torch.Tensor, k_xmax: torch.Tensor, k_xmin: torch.Tensor, block_size: int, top_pct: float = 0.10) -> Tuple[torch.Tensor, np.ndarray]:
    q_xmx = q_xmax.view(-1).cpu()
    q_xmn = q_xmin.view(-1).cpu()
    k_xmx = k_xmax.view(-1).cpu()
    k_xmn = k_xmin.view(-1).cpu()
    
    D = q_xmx.numel()
    assert D == q_xmn.numel() == k_xmx.numel() == k_xmn.numel()
    assert D % block_size == 0
    K = D // block_size
    
    # 四元数特征：q_xmax, q_xmin, k_xmax, k_xmin
    quadruple_features = torch.stack([q_xmx, q_xmn, k_xmx, k_xmn], dim=1)  # [D, 4]
    
    # 第一阶段：交替排序策略
    # 单数block按q的大小排序，双数block按k的大小排序
    q_abs_max = torch.max(torch.abs(q_xmx), torch.abs(q_xmn))  # [D]
    k_abs_max = torch.max(torch.abs(k_xmx), torch.abs(k_xmn))  # [D]
    
    # 创建交替排序的分数
    alternating_scores = torch.zeros_like(q_abs_max)
    for i in range(D):
        block_idx = i // block_size
        if block_idx % 2 == 0:  # 双数block (0, 2, 4, ...) 按k排序
            alternating_scores[i] = k_abs_max[i]
        else:  # 单数block (1, 3, 5, ...) 按q排序
            alternating_scores[i] = q_abs_max[i]
    
    idx_abs_sorted = torch.argsort(alternating_scores, descending=True, stable=True)
    blocks = idx_abs_sorted.view(K, block_size)
    
    top_pct = float(max(0.0, min(0.5, top_pct)))
    keep = int(round(K * top_pct))
    
    if keep == 0:
        # 使用四元数进行K-means聚类
        idx_mid, counts_mid = quadruple_kmeans_calc_reorder_index(quadruple_features, K, block_size)
        return idx_mid.to(torch.long), counts_mid
    
    K_rem = K - 2 * keep
    if K_rem <= 0:
        return idx_abs_sorted.to(torch.long), np.full((K,), block_size)
    
    left_keep = blocks[:keep].reshape(-1)
    right_keep = blocks[K - keep:].reshape(-1)
    mid_idx = blocks[keep:K - keep].reshape(-1)
    
    # 对中间部分使用四元数K-means聚类
    mid_quadruple = quadruple_features[mid_idx]
    idx_mid_local, counts_mid = quadruple_kmeans_calc_reorder_index(mid_quadruple, K_rem, block_size)
    mid_ordered = mid_idx[idx_mid_local]
    
    final_idx = torch.cat([left_keep, mid_ordered, right_keep], dim=0).to(torch.long)
    counts = np.hstack([np.full((keep,), block_size), counts_mid, np.full((keep,), block_size)])
    return final_idx, counts


def quadruple_kmeans_calc_reorder_index(quadruple_features: torch.Tensor, n_clusters: int, block_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    """使用四元数特征进行K-means聚类"""
    D, _ = quadruple_features.shape
    assert D % block_size == 0
    K = D // block_size
    assert K == n_clusters
    
    data_np = quadruple_features.numpy()
    
    # 对四元数特征进行标准化
    if data_np.shape[0] > 1000:
        data_mean = np.mean(data_np, axis=0)
        data_std = np.std(data_np, axis=0) + 1e-8
        data_np = (data_np - data_mean) / data_std
        print(f"Applied normalization for quadruple features with {data_np.shape[0]} channels")
    
    def _greedy_capacity_4d(data_np: np.ndarray, K: int, cap_size: int):
        D_head = data_np.shape[0]
        assert D_head == K * cap_size
        # 针对四元数特征优化K-means参数
        if D_head > 1000:
            # 大量通道：使用K-means++初始化，增加迭代次数
            km = KMeans(n_clusters=K, n_init=20, random_state=0, init='k-means++', max_iter=500).fit(data_np)
        else:
            # 少量通道：使用默认参数
            km = KMeans(n_clusters=K, n_init=10, random_state=0, max_iter=300).fit(data_np)
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
    
    # 直接使用sklearn的KMeans
    assign_np, centers_t, dists = _greedy_capacity_4d(data_np, K, block_size)
    labels = torch.from_numpy(assign_np)
    
    centers_center = centers_t.mean(dim=1)
    cluster_order = torch.argsort(centers_center).cpu().numpy().tolist()
    dnp = dists.cpu().numpy()
    idx_list = []
    counts = []
    for k in cluster_order:
        members = torch.nonzero(labels == k, as_tuple=False).view(-1).cpu().numpy()
        members = members[np.argsort(dnp[members, k])]
        idx_list.append(torch.from_numpy(members))
        counts.append(len(members))
    
    all_index = torch.hstack(idx_list)
    all_counts = np.hstack(counts)
    return all_index, all_counts


@torch.no_grad()
def compute_reorders(oc_stats: Dict[str, Tuple[torch.Tensor, torch.Tensor]], block_size: int, method: str = 'peg', interval_key: str = 'center', acts_for_mse: Dict[str, torch.Tensor] | None = None, sls_iters: int = 20, sls_block_axis: str = 'feature', sls_scale_format: str = 'e4m3', hybrid_top_pct: float = 0.10) -> Dict[str, torch.Tensor]:
    perms: Dict[str, torch.Tensor] = {}
    
    # 特殊处理 q_proj 和 k_proj 的情况
    q_proj_key = None
    k_proj_key = None
    
    # 查找 q_proj 和 k_proj 的键
    for name in oc_stats.keys():
        if name.endswith('q_proj'):
            q_proj_key = name
        elif name.endswith('k_proj'):
            k_proj_key = name
    
    # 如果找到了 q_proj 和 k_proj，使用四元数方法
    if q_proj_key is not None and k_proj_key is not None:
        q_xmax, q_xmin = oc_stats[q_proj_key]
        k_xmax, k_xmin = oc_stats[k_proj_key]
        
        D = q_xmax.numel()
        if D % block_size == 0 and q_xmax.numel() == k_xmax.numel():
            print(f"Using quadruple reordering for {q_proj_key} and {k_proj_key}")
            idx, counts = qk_proj_quadruple_reorder_index(q_xmax, q_xmin, k_xmax, k_xmin, block_size, top_pct=hybrid_top_pct)
            if idx.numel() == D:
                perms[q_proj_key] = idx.to(torch.long)
                perms[k_proj_key] = idx.to(torch.long)  # 使用相同的重排序顺序
    
    # 处理其他模块
    for name, (xmax, xmin) in oc_stats.items():
        # 跳过已经处理过的 q_proj 和 k_proj
        if name == q_proj_key or name == k_proj_key:
            continue
            
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
