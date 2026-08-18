import torch
import torch.nn.functional as F


def _pairwise_sqdist(x):
    # x: (N, C)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    dist = xx + xx.t() - 2 * x @ x.t() #dist=||(x1,y1,z1),(x2,y2,z2)||2
    return dist.clamp(min=0.0)


def _knn_idx(x, k=16):
    # x: (N, C)
    n = x.shape[0]
    if n <= 1:
        return torch.zeros((n, 1), dtype=torch.long, device=x.device)
    k = min(k, n - 1)
    dist = _pairwise_sqdist(x)
    _, idx = torch.topk(dist, k=k + 1, dim=1, largest=False)
    return idx[:, 1:]

def build_boundary_mask(points_xyz, ins_label, sem_label=None, k=16, ratio_thresh=0.3):
    """
    根据 GT instance 邻域差异构造 boundary mask
    points_xyz: (B, N, 3)
    ins_label : (B, N)
    sem_label : (B, N) or None
    return    : (B, N) bool
    """
    B, N, _ = points_xyz.shape
    out = []

    for b in range(B):
        pts = points_xyz[b]
        ins = ins_label[b].reshape(-1).long()

        if sem_label is not None:
            sem = sem_label[b].reshape(-1).long()
            valid_mask = (ins >= 0) & (sem != 2)
        else:
            valid_mask = (ins >= 0)

        boundary = torch.zeros(N, dtype=torch.bool, device=pts.device)

        valid_idx = torch.where(valid_mask)[0]
        if valid_idx.numel() <= 1:
            out.append(boundary)
            continue

        pts_valid = pts[valid_idx]
        ins_valid = ins[valid_idx]
        nn_idx = _knn_idx(pts_valid, k=min(k, max(1, pts_valid.shape[0] - 1))) #knn

        nbr_ins = ins_valid[nn_idx]                          # (Nvalid, k)
        self_ins = ins_valid.unsqueeze(1).expand_as(nbr_ins)
        diff_ratio = (nbr_ins != self_ins).float().mean(dim=1)
        boundary_valid = diff_ratio > ratio_thresh
        boundary[valid_idx] = boundary_valid
        out.append(boundary)

    return torch.stack(out, dim=0)


def discriminative_embedding_loss(point_feature,
                                  ins_label,
                                  sem_label=None,
                                  boundary_mask=None,
                                  delta_var=0.5,
                                  delta_dist=1.5,
                                  weight_boundary=2.0):
    """
    point_feature: (B, N, C)
    ins_label    : (B, N)
    sem_label    : (B, N)
    boundary_mask: (B, N) bool
    """
    B, N, C = point_feature.shape
    total_var = point_feature.new_tensor(0.0)
    total_dist = point_feature.new_tensor(0.0)
    total_reg = point_feature.new_tensor(0.0)
    valid_batch = 0

    feat = F.normalize(point_feature, p=2, dim=-1)

    for b in range(B):
        emb = feat[b]
        ins = ins_label[b].reshape(-1).long()

        if sem_label is not None:
            sem = sem_label[b].reshape(-1).long()
            valid_mask = (ins >= 0) & (sem == 1)
        else:
            valid_mask = (ins >= 0)

        emb = emb[valid_mask]
        ins = ins[valid_mask]

        if boundary_mask is not None:
            bmask = boundary_mask[b].reshape(-1)[valid_mask].float()
        else:
            bmask = torch.zeros_like(ins).float()

        unique_ids = torch.unique(ins)
        unique_ids = unique_ids[unique_ids >= 0]

        if unique_ids.numel() == 0:
            continue

        centroids = []
        var_loss = emb.new_tensor(0.0)

        for uid in unique_ids:
            mask = (ins == uid)
            pts_i = emb[mask]
            if pts_i.shape[0] == 0:
                continue

            w = 1.0 + weight_boundary * bmask[mask]
            centroid = (pts_i * w.unsqueeze(1)).sum(dim=0) / (w.sum() + 1e-6)
            centroids.append(centroid)

            dist = torch.norm(pts_i - centroid.unsqueeze(0), dim=1)
            dist = F.relu(dist - delta_var) ** 2
            dist = dist.mean()
            var_loss += dist

        centroids = torch.stack(centroids, dim=0)   # (K, C)
        var_loss = var_loss / max(1, len(centroids))

        if centroids.shape[0] > 1:
            cdist = torch.cdist(centroids, centroids, p=2)
            cdist = torch.unique(cdist)
            cdist = [element for element in cdist if element != 0]
            cdist = torch.tensor(cdist)
            dist_loss = F.relu(delta_dist - cdist) ** 2
            dist_loss = dist_loss.mean()
        else:
            dist_loss = emb.new_tensor(0.0)

        reg_loss = torch.norm(centroids, dim=1).mean()

        total_var += var_loss
        total_dist += dist_loss
        total_reg += reg_loss
        valid_batch += 1

    if valid_batch == 0:
        return point_feature.new_tensor(0.0)

    total_var = total_var / valid_batch
    total_dist = total_dist / valid_batch
    total_reg = total_reg / valid_batch

    return  total_var + total_dist + 0.001 * total_reg


def plane_consistency_loss(points_xyz, point_offset, ins_label, sem_label=None, min_pts=20):
    """
    使用 offset 修正后的点，按 GT instance 做平面一致性约束
    points_xyz   : (B, N, 3)
    point_offset : (B, N, 3)
    ins_label    : (B, N)
    """
    corrected = points_xyz + point_offset
    B, N, _ = corrected.shape
    total_loss = corrected.new_tensor(0.0)
    total_count = 0

    for b in range(B):
        pts = corrected[b]
        ins = ins_label[b].reshape(-1).long()

        if sem_label is not None:
            sem = sem_label[b].reshape(-1).long()
            valid_mask = (ins >= 0) & (sem == 1)
        else:
            valid_mask = (ins >= 0)

        pts = pts[valid_mask]
        ins = ins[valid_mask]

        unique_ids = torch.unique(ins)
        unique_ids = unique_ids[unique_ids >= 0]

        for uid in unique_ids:
            mask = (ins == uid)
            pts_i = pts[mask]

            if pts_i.shape[0] < min_pts:
                continue

            mean = pts_i.T.mean(dim=1, keepdim=True)
            pts_c = pts_i.T - mean
            cov = 1000 * torch.cov(pts_c)

            eigvals, eigvecs = torch.linalg.eigh(cov)
            planarity = min(eigvals)
            # normal = eigvecs[:, 0]
            # residual = torch.abs(pts_c @ normal).sum()
            #
            # planarity = eigvals[0] / (eigvals.sum() + 1e-6)
            total_loss += planarity
            total_count += 1

    if total_count == 0:
        return corrected.new_tensor(0.0)

    return total_loss / total_count

def plane_consistency_loss_s(points_xyz, point_offset, ins_label, sem_label=None, min_pts=20):
    """
    使用 offset 修正后的点，按 GT instance 做平面一致性约束
    points_xyz   : (B, N, 3)
    point_offset : (B, N, 3)
    ins_label    : (B, N)
    """
    corrected = points_xyz + point_offset
    B, N, _ = corrected.shape
    total_loss = corrected.new_tensor(0.0)
    total_count = 0

    for b in range(B):
        pts = corrected[b]
        ins = ins_label[b].reshape(-1).long()

        if sem_label is not None:
            sem = sem_label[b].reshape(-1).long()
            valid_mask = (ins >= 0) & (sem != 2)
        else:
            valid_mask = (ins >= 0)

        pts = pts[valid_mask]
        ins = ins[valid_mask]

        unique_ids = torch.unique(ins)
        unique_ids = unique_ids[unique_ids >= 0]

        for uid in unique_ids:
            mask = (ins == uid)
            pts_i = pts[mask]

            if pts_i.shape[0] < min_pts:
                continue

            center = pts_i.mean(dim=0, keepdim=True) #局部邻域点（平面）质心
            pts_c = pts_i - center #平面点云去中心化
            cov = pts_c.t() @ pts_c / (pts_i.shape[0] + 1e-6) #计算去中心化后点云的3*3协方差矩阵

            eigvals, eigvecs = torch.linalg.eigh(cov) #解算特征值、特征向量
            normal = eigvecs[:, 0] #法向量
            residual = torch.abs(pts_c @ normal).mean() #计算点云到平面的平均绝对距离（残差），
                                                        #通过计算每个点在法向量方向上的投影长度，
                                                        #因为点云已经过中心化，所以这个投影值直接反映了点到平面的带符号距离，
                                                        #.abs()保留绝对几何距离，
                                                        #.mean()计算所有点到该平面绝对距离的平均值，作为评估平面拟合好坏的损失函数（Loss）或残差。该值越接近 0，说明点云越接近一个完美的平面。

            planarity = eigvals[0] / (eigvals.sum() + 1e-6) #邻域内表面变化率
            total_loss += residual + 0.1 * planarity
            total_count += 1

    if total_count == 0:
        return corrected.new_tensor(0.0)

    return total_loss / total_count