import os
import queue
import torch
import numpy as np
import torch.nn.functional as F

from sklearn.cluster import DBSCAN

from .model_utils import *

def ensure_result_dirs():
    sub_dirs = [
        'result',
        'result/offset',
        'result/feature',
        'result/feature_norm',
        'result/label',
        'result/sem_pred',
        'result/cluster_init',
        'result/cluster_final',
        'result/cluster_edge',
        'result/pred_final'
    ]
    for d in sub_dirs:
        os.makedirs(d, exist_ok=True)


def save_final_prediction(frame_id, xyz, sem_gt, ins_gt, sem_pred, ins_pred):
    save_path = 'result/pred_final/' + str(frame_id[0])[:-4] + '.txt'

    xyz_np = xyz.detach().cpu().numpy().squeeze()
    sem_gt_np = sem_gt.detach().cpu().numpy().squeeze().reshape(-1, 1)
    ins_gt_np = ins_gt.detach().cpu().numpy().squeeze().reshape(-1, 1)
    sem_pred_np = sem_pred.detach().cpu().numpy().squeeze().reshape(-1, 1)
    ins_pred_np = ins_pred.detach().cpu().numpy().squeeze().reshape(-1, 1)

    out = np.concatenate(
        [xyz_np, sem_gt_np, ins_gt_np, sem_pred_np, ins_pred_np],
        axis=1
    )
    np.savetxt(save_path, out, fmt='%.6f', delimiter=' ')


def _get_roof_confidence(batch_dict):
    """
    point_pred_score: (1, N, num_classes)
    约定 roof 类索引为 1
    """
    if 'point_pred_score' not in batch_dict:
        return None
    score = batch_dict['point_pred_score']
    if score.dim() != 3:
        return None
    if score.shape[-1] < 2:
        return None
    return score[0, :, 1]


def _merge_small_clusters(cluster_idx, coords):
    """
    小簇并到最近大簇，避免过分裂
    cluster_idx: (N,)
    xyz: (N, 3)
    fea: (N, C)
    """
    uniq = [int(x) for x in torch.unique(cluster_idx).tolist() if x >= 0]
    if len(uniq) == 0:
        return cluster_idx

    plane_n = []
    plane_D = []
    D = []
    
    for cid in uniq:
        m = (cluster_idx == cid)
        coords_c = coords[m].mean(dim=0, keepdim=True)
        C = coords[m]-coords_c
        _, eigvecs = torch.linalg.eigh(torch.cov(C.T))
        plane_n.append(eigvecs[:, 0] / torch.norm(eigvecs[:, 0]))
        D.append(torch.abs(coords_c @ eigvecs[:, 0]))

    D = torch.stack(D).squeeze()
    plane_n = torch.stack(plane_n)
    for cid in uniq:
        plane_D.append(D-D[cid])

    plane_D = torch.stack(plane_D).detach().cpu().numpy()
    plane_M = torch.abs(plane_n @ plane_n.T).detach().cpu().numpy()
    D = np.triu(plane_D, k=1)
    C = np.triu(plane_M, k=1)
    Q=[]

    for cid in uniq:
        if cid in Q:
            continue
        else:
            m = (np.abs(C[cid,:] - 1) < 0.01) & (np.abs(D[cid, :]) < 0.1)
            n = np.where(m == True)[0]
            Q.extend(n)
            for tgt in n:
                cluster_idx[cluster_idx == tgt] = cid


    # 重映射成连续标签
    uniq2 = [int(x) for x in torch.unique(cluster_idx).tolist() if x >= 0]
    remap = {old: new for new, old in enumerate(sorted(uniq2))}
    for old, new in remap.items():
        cluster_idx[cluster_idx == old] = new

    return cluster_idx


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


def Cluster_all(batch_dict):
    ensure_result_dirs()

    # frame_id = batch_dict['frame_id'][0].split('\\')[-1]
    frame_id = batch_dict['frame_id']
    coords = batch_dict['coords']
    xyz = batch_dict['xyz']                    # (1, N, 3)
    pts_sem = batch_dict['point_pred_sem']    # (1, N)
    sem_label = batch_dict['sem_label']       # (1, N)
    ins_label = batch_dict['ins_label']       # (1, N)

    offset = batch_dict['point_pred_offset']  # (1, N, 3)
    offset_pts = xyz.clone() + offset         # (1, N, 3)

    pts_fea = batch_dict['point_feature']     # (1, N, C)
    pts_fea = F.normalize(pts_fea, p=2, dim=2, eps=1e-12)

    sem_labels = pts_sem.squeeze(0)           # (N,)
    xyz_s = offset_pts.squeeze(0)             # (N, 3)
    fea_s = pts_fea.squeeze(0)                # (N, C)
    N = sem_labels.size(0)
    coords = coords.squeeze(0)
    ins_label = ins_label.squeeze(0).squeeze(0)

    d_coords = torch.cdist(coords, coords, p=2)
    _, index = torch.topk(d_coords, k=9, largest=False)
    knn_coords = coords[index[:,1:]]
    mean_coords = knn_coords.mean(dim=1, keepdim=True) + 1e-6
    norm_coords = knn_coords - mean_coords
    normal = []
    point_d = []

    for i in range(norm_coords.shape[0]):
        C = norm_coords[i,:].squeeze(0)
        eigvals, eigvecs = torch.linalg.eigh(torch.cov(C.T))
        normal.append(eigvecs[:, 0] / torch.norm(eigvecs[:, 0]))
        point_d.append(torch.abs(mean_coords[i,:] @ eigvecs[:, 0]))

    normal = torch.stack(normal)
    matrix = torch.abs(normal @ normal.T)


    # --------------------------------------
    # nc feature clustering
    # --------------------------------------
    Radius = 0.2
    cluster_ptnum_thresh = 15
    sem_labels = pts_sem.squeeze()
    clusters = []
    v = sem_labels.new_zeros(sem_labels.shape)
    for i in range(N):
        # if seed_mask[i] != True:
        if sem_labels[i] != True:
                v[i] = 1
    for i in range(N):
        if v[i] == 0:
            Q = queue.Queue()
            cluster_C = []
            v[i] = 1
            Q.put(i)
            cluster_C.append(i)
            if ~Q.empty():
                k = Q.get()
                for j in range(N):
                    r1 = torch.dist(xyz_s[j, :], xyz_s[k, :])
                    r2 = torch.dist(fea_s[j, :], fea_s[k, :])
                    r = 0.05 * r1 + 0.95 * r2
                    if (r < Radius) & (matrix[j,k] > 0.7):
                        if sem_labels[j] == sem_labels[k] and v[j] == 0:
                            v[j] = 1
                            Q.put(j)
                            cluster_C.append(j)
            if len(cluster_C) > cluster_ptnum_thresh:
                clusters.append(cluster_C)
    cluster_idx = sem_labels.new_ones(sem_labels.shape) * -1
    for x in range(len(clusters)):
        for y in range(len(clusters[x])):
            cluster_idx[clusters[x][y]] = x

    # --------------------------------------
    # cluster merging
    # --------------------------------------
    cluster_idx = _merge_small_clusters(cluster_idx, coords)

    # --------------------------------------
    # point reassignment
    # --------------------------------------
    uniq = [int(x) for x in torch.unique(cluster_idx).tolist() if x >= 0]

    if len(uniq) > 0:
        planes = []
        xyz_centers = []
        fea_centers = []

        for cid in uniq:
            m = (cluster_idx == cid)
            pts_np = xyz_s[m].detach().cpu().numpy()
            a, b, c, d = planefit(pts_np, True)
            planes.append((a, b, c, d))
            xyz_centers.append(xyz_s[m].mean(dim=0))
            fea_centers.append(F.normalize(fea_s[m].mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0))

        xyz_centers = torch.stack(xyz_centers, dim=0)
        fea_centers = torch.stack(fea_centers, dim=0)

        reassign_mask = (cluster_idx < 0) & (sem_labels != 2)
        reassign_idx = torch.where(reassign_mask)[0]

        for idx_pt in reassign_idx:
            p_xyz = xyz_s[idx_pt].detach().cpu().numpy()
            p_fea = fea_s[idx_pt].unsqueeze(0)

            d_plane = []
            for pl in planes:
                d_plane.append(point_plane_dist(p_xyz, pl))
            d_plane = np.array(d_plane, dtype=np.float32)

            d_xyz2c = torch.cdist(xyz_s[idx_pt].unsqueeze(0), xyz_centers, p=2).squeeze(0).detach().cpu().numpy()
            d_fea2c = torch.cdist(p_fea, fea_centers, p=2).squeeze(0).detach().cpu().numpy()

            d_plane = d_plane / (d_plane.mean() + 1e-6) #1
            d_xyz2c = d_xyz2c / (d_xyz2c.mean() + 1e-6)
            d_fea2c = d_fea2c / (d_fea2c.mean() + 1e-6)

            d = 0.50 * d_plane + 0.25 * d_xyz2c + 0.25 * d_fea2c
            tgt = int(np.argmin(d))
            cluster_idx[idx_pt] = tgt


    cluster_idx[sem_labels == 2] = -1

    a = [int(x) for x in torch.unique(cluster_idx).tolist() if x >= 0]
    b = [int(x) for x in torch.unique(ins_label).tolist() if x >= 0]
    if len(a)<len(b) :
        print(frame_id)

    batch_dict['point_pred_ins'] = cluster_idx.unsqueeze(0)
    xyz_np = xyz.detach().cpu().numpy().squeeze()
    cluster_np = cluster_idx.detach().cpu().numpy().reshape(-1, 1)

    cluster_init_path = 'result/cluster_init/' + str(frame_id[0])[:-4] + '.txt'
    np.savetxt(cluster_init_path, np.concatenate((xyz_np, cluster_np), axis=1), fmt='%.6f', delimiter=' ')

    cluster_final_path = 'result/cluster_final/' + str(frame_id[0])[:-4] + '.txt'
    np.savetxt(cluster_final_path, np.concatenate((xyz_np, cluster_np), axis=1), fmt='%.6f', delimiter=' ')

    save_final_prediction(
        frame_id=frame_id,
        xyz=xyz,
        sem_gt=sem_label,
        ins_gt=ins_label,
        sem_pred=pts_sem,
        ins_pred=cluster_idx.unsqueeze(0)
    )