from .pointnet2 import PointNet2
import torch.nn as nn
from .ncpointgrouping import Cluster_all
from .innovation_losses import (
    build_boundary_mask,
    discriminative_embedding_loss,
    plane_consistency_loss_s
)


class Net(nn.Module):
    def __init__(self, model_cfg, input_channel=3, cluster=False):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = PointNet2(model_cfg.PointNet2, input_channel)
        self.cluster = cluster

        # 这两个权重你后面可以调
        self.w_embed = 0.20
        self.w_plane = 0.50

    def forward(self, batch_dict):
        batch_dict = self.backbone(batch_dict)

        if self.training:
            loss = 0
            loss_dict = {}
            disp_dict = {}

            # backbone 原始损失
            tmp_loss, loss_dict, disp_dict = self.backbone.loss(loss_dict, disp_dict)
            loss += tmp_loss

            # ===== 创新1：边界感知判别式 embedding loss =====
            # 这里优先用 xyz（归一化坐标），没有就退回 points 的前三维
            if 'xyz' in batch_dict:
                points_xyz = batch_dict['xyz']
            else:
                points_xyz = batch_dict['points'][..., :3]

            boundary_mask = build_boundary_mask(
                points_xyz=points_xyz,
                ins_label=batch_dict['ins_label'],
                sem_label=batch_dict.get('sem_label', None),
                k=16,
                ratio_thresh=0.3
            )

            emb_loss = discriminative_embedding_loss(
                point_feature=batch_dict['point_feature'],
                ins_label=batch_dict['ins_label'],
                sem_label=batch_dict.get('sem_label', None),
                boundary_mask=boundary_mask,
                delta_var=0.5,
                delta_dist=2.0,
                weight_boundary=2.0
            )

            loss += self.w_embed * emb_loss
            loss_dict['loss_embed'] = emb_loss.item()
            disp_dict['loss_embed'] = round(emb_loss.item(), 4)

            # ===== 创新2：平面几何一致性损失 =====
            plane_loss = plane_consistency_loss_s(
                points_xyz=points_xyz,
                point_offset=batch_dict['point_pred_offset'],
                ins_label=batch_dict['ins_label'],
                sem_label=batch_dict.get('sem_label', None),
                min_pts=20
            )

            loss += plane_loss
            loss_dict['loss_plane'] = plane_loss.item()
            disp_dict['loss_plane'] = round(plane_loss.item(), 4)

            return loss, loss_dict, disp_dict

        else:
            if not self.cluster:
                Cluster_edge(batch_dict)
            else:
                Cluster_all(batch_dict)
            return batch_dict