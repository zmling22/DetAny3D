import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import LayerNorm2d, MLPBlock
from typing import Dict, List, Optional, Tuple, Union
import math

class Detector2DHead(nn.Module):
    def __init__(self, cfg, num_anchors: int = 1, box_dim: int = 2, conv_dims: List[int] = (-1,)):
    
        super(Detector2DHead, self).__init__()
        
        self.num_classes = 2
        self.top_k = 30

        cur_channels = cfg.dino_dim
        # Keeping the old variable names and structure for backwards compatiblity.
        # Otherwise the old checkpoints will fail to load.
        if len(conv_dims) == 1:
            out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
            # 3x3 conv for the hidden representation
            self.conv = self._get_rpn_conv(cur_channels, out_channels)
            cur_channels = out_channels
        else:
            self.conv = nn.Sequential()
            for k, conv_dim in enumerate(conv_dims):
                out_channels = cur_channels if conv_dim == -1 else conv_dim
                if out_channels <= 0:
                    raise ValueError(
                        f"Conv output channels should be greater than 0. Got {out_channels}"
                    )
                conv = self._get_rpn_conv(cur_channels, out_channels)
                self.conv.add_module(f"conv{k}", conv)
                cur_channels = out_channels
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

        # Keeping the order of weights initialization same for backwards compatiblility.
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def _get_rpn_conv(self, in_channels, out_channels):
        return Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=nn.ReLU(),
        )
    
    def forward(self, dino_features, vit_pad_size, gt_masks=None):
        """
        前向传播：预测每个 patch 的类别和分数。
        Args:
            x (Tensor): 输入图像，形状为 (B, C, H, W)。
            gt_masks (Tensor): 形状为 (B, num_objects, H, W)，每个物体的 mask。
        
        Returns:
            point_prompts (Tensor): 预测的 point prompt，形状为 (B, 2)，每个物体的中心点偏移量。
            loss (Tensor): 损失函数值。
        """

        pred_objectness_logits = []
        pred_anchor_deltas = []
        # import ipdb;ipdb.set_trace()
        for x in dino_features:
            x = x.permute(0, 3, 1, 2)
            t = self.conv(x)
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        
        # # 对每个 patch 预测类别和分数
        # # import ipdb;ipdb.set_trace()
        # dino_features = self.dino_neck(dino_features.permute(0, 3, 1, 2))
        # dino_features = dino_features.permute(0, 2, 3, 1)
        

        # logits = self.cls_head(dino_features).flatten(1, 2)  # 形状为 (B, N, num_classes)
        # logits = F.softmax(logits, dim=1)  # 在类别维度上计算 softmax (B, N, num_classes)

        # 获取分数最大的 top_k 个 patch
        scores_with_indices = torch.topk(pred_objectness_logits[-1].flatten(2), self.top_k, dim=-1)  # 形状为 (B, top_k)
        top_k_scores = scores_with_indices.values
        top_k_indices = scores_with_indices.indices
        
        # 计算 top_k patch 的中心点
        patch_num_H, patch_num_W = dino_features[-1].shape[1], dino_features[-1].shape[2]

        # 将 flat index 转为 2D index (row, col)
        y_coords = top_k_indices // patch_num_W  # 计算每个 patch 的行索引 (B, top_k)
        x_coords = top_k_indices % patch_num_W 

        # 计算中心点坐标
        point_prompts = torch.stack([(y_coords + 0.5) * 14, (x_coords + 0.5) * 14], dim=-1)  # (B, top_k, 2)
        point_prompts = point_prompts.squeeze(0).squeeze(0).long()
        
        # 计算损失
        if gt_masks is not None:
            loss = self.compute_loss(point_prompts, gt_masks) / gt_masks.shape[0]
        else:
            loss = None
            
        return point_prompts, loss
    
    def compute_loss(self, point_prompts, gt_masks):
        loss = torch.tensor(0.0, device=point_prompts.device)  # 初始化为 Tensor 类型，确保设备一致
        
        # 遍历所有的物体
        for obj_idx in range(gt_masks.shape[0]):
            mask = gt_masks[obj_idx]  # 获取当前物体的 mask
            
            if mask.sum() == 0:  # 如果该物体没有在 GT 中出现
                continue
            
            # 检查 point_prompts 是否在该物体的 mask 区域内
            # point_prompts 是 (B, top_k, 2)，所以我们需要检查点是否在相应的 mask 区域内
            prompt_y = point_prompts[:, 0]  # 取出所有点的 y 坐标
            prompt_x = point_prompts[:, 1]  # 取出所有点的 x 坐标
            
            if mask[prompt_y, prompt_x].sum() == 0:
                loss += 1.0  # 可以根据需要调整惩罚系数
                    
        return loss

# class StandardRPNHead(nn.Module):
#     """
#     Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
#     Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
#     objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
#     specifying how to deform each anchor into an object proposal.
#     """

#     def __init__(
#         self, cfg, num_anchors: int = 1, box_dim: int = 2, conv_dims: List[int] = (-1,)
#     ):
#         """
#         NOTE: this interface is experimental.

#         Args:
#             in_channels (int): number of input feature channels. When using multiple
#                 input features, they must have the same number of channels.
#             num_anchors (int): number of anchors to predict for *each spatial position*
#                 on the feature map. The total number of anchors for each
#                 feature map will be `num_anchors * H * W`.
#             box_dim (int): dimension of a box, which is also the number of box regression
#                 predictions to make for each anchor. An axis aligned box has
#                 box_dim=4, while a rotated box has box_dim=5.
#             conv_dims (list[int]): a list of integers representing the output channels
#                 of N conv layers. Set it to -1 to use the same number of output channels
#                 as input channels.
#         """
#         super().__init__()
#         cur_channels = cfg.dino_dim
#         # Keeping the old variable names and structure for backwards compatiblity.
#         # Otherwise the old checkpoints will fail to load.
#         if len(conv_dims) == 1:
#             out_channels = cur_channels if conv_dims[0] == -1 else conv_dims[0]
#             # 3x3 conv for the hidden representation
#             self.conv = self._get_rpn_conv(cur_channels, out_channels)
#             cur_channels = out_channels
#         else:
#             self.conv = nn.Sequential()
#             for k, conv_dim in enumerate(conv_dims):
#                 out_channels = cur_channels if conv_dim == -1 else conv_dim
#                 if out_channels <= 0:
#                     raise ValueError(
#                         f"Conv output channels should be greater than 0. Got {out_channels}"
#                     )
#                 conv = self._get_rpn_conv(cur_channels, out_channels)
#                 self.conv.add_module(f"conv{k}", conv)
#                 cur_channels = out_channels
#         # 1x1 conv for predicting objectness logits
#         self.objectness_logits = nn.Conv2d(cur_channels, num_anchors, kernel_size=1, stride=1)
#         # 1x1 conv for predicting box2box transform deltas
#         self.anchor_deltas = nn.Conv2d(cur_channels, num_anchors * box_dim, kernel_size=1, stride=1)

#         # Keeping the order of weights initialization same for backwards compatiblility.
#         for layer in self.modules():
#             if isinstance(layer, nn.Conv2d):
#                 nn.init.normal_(layer.weight, std=0.01)
#                 nn.init.constant_(layer.bias, 0)

#     def _get_rpn_conv(self, in_channels, out_channels):
#         return Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             activation=nn.ReLU(),
#         )

#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         # Standard RPN is shared across levels:
#         in_channels = [s.channels for s in input_shape]
#         assert len(set(in_channels)) == 1, "Each level must have the same channel!"
#         in_channels = in_channels[0]

#         # RPNHead should take the same input as anchor generator
#         # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
#         anchor_generator = build_anchor_generator(cfg, input_shape)
#         num_anchors = anchor_generator.num_anchors
#         box_dim = anchor_generator.box_dim
#         assert (
#             len(set(num_anchors)) == 1
#         ), "Each level must have the same number of anchors per spatial position"
#         return {
#             "in_channels": in_channels,
#             "num_anchors": num_anchors[0],
#             "box_dim": box_dim,
#             "conv_dims": cfg.MODEL.RPN.CONV_DIMS,
#         }

#     def forward(self, features: List[torch.Tensor]):
#         """
#         Args:
#             features (list[Tensor]): list of feature maps

#         Returns:
#             list[Tensor]: A list of L elements.
#                 Element i is a tensor of shape (N, A, Hi, Wi) representing
#                 the predicted objectness logits for all anchors. A is the number of cell anchors.
#             list[Tensor]: A list of L elements. Element i is a tensor of shape
#                 (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
#                 to proposals.
#         """
#         pred_objectness_logits = []
#         pred_anchor_deltas = []
#         import ipdb;ipdb.set_trace()
#         for x in features:
#             x = x.permute(0, 3, 1, 2)
#             t = self.conv(x)
#             pred_objectness_logits.append(self.objectness_logits(t))
#             pred_anchor_deltas.append(self.anchor_deltas(t))
#         return pred_objectness_logits, pred_anchor_deltas


# class RPN(nn.Module):
#     """
#     Region Proposal Network, introduced by :paper:`Faster R-CNN`.
#     """

#     def __init__(
#         self,
#         cfg,
#         head: nn.Module,
#         anchor_generator: nn.Module,
#         anchor_matcher,
#         box2box_transform,
#         batch_size_per_image: int,
#         positive_fraction: float,
#         pre_nms_topk: Tuple[float, float],
#         post_nms_topk: Tuple[float, float],
#         nms_thresh: float = 0.7,
#         min_box_size: float = 0.0,
#         anchor_boundary_thresh: float = -1.0,
#         loss_weight: Union[float, Dict[str, float]] = 1.0,
#         box_reg_loss_type: str = "smooth_l1",
#         smooth_l1_beta: float = 0.0,
#     ):
#         """
#         NOTE: this interface is experimental.

#         Args:
#             in_features (list[str]): list of names of input features to use
#             head (nn.Module): a module that predicts logits and regression deltas
#                 for each level from a list of per-level features
#             anchor_generator (nn.Module): a module that creates anchors from a
#                 list of features. Usually an instance of :class:`AnchorGenerator`
#             anchor_matcher (Matcher): label the anchors by matching them with ground truth.
#             box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
#                 instance boxes
#             batch_size_per_image (int): number of anchors per image to sample for training
#             positive_fraction (float): fraction of foreground anchors to sample for training
#             pre_nms_topk (tuple[float]): (train, test) that represents the
#                 number of top k proposals to select before NMS, in
#                 training and testing.
#             post_nms_topk (tuple[float]): (train, test) that represents the
#                 number of top k proposals to select after NMS, in
#                 training and testing.
#             nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
#             min_box_size (float): remove proposal boxes with any side smaller than this threshold,
#                 in the unit of input image pixels
#             anchor_boundary_thresh (float): legacy option
#             loss_weight (float|dict): weights to use for losses. Can be single float for weighting
#                 all rpn losses together, or a dict of individual weightings. Valid dict keys are:
#                     "loss_rpn_cls" - applied to classification loss
#                     "loss_rpn_loc" - applied to box regression loss
#             box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
#             smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
#                 use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
#         """
#         super().__init__()
#         self.rpn_head = StandardRPNHead(cfg)
#         self.anchor_generator = DefaultAnchorGenerator()
#         self.anchor_matcher = Matcher(
#             [0.3, 0.7], [0, -1, 1], allow_low_quality_matches=True
#         )
#         self.box2box_transform = Box2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
#         self.batch_size_per_image = 1
#         self.positive_fraction = 0.5
#         # Map from self.training state to train/test settings
#         self.pre_nms_topk = {True: 1200, False: 600}
#         self.post_nms_topk = {True: 50, False: 30}
#         self.nms_thresh = 0.7
#         self.min_box_size = float(min_box_size)
#         self.anchor_boundary_thresh = anchor_boundary_thresh
#         if isinstance(loss_weight, float):
#             loss_weight = {"loss_rpn_cls": loss_weight, "loss_rpn_loc": loss_weight}
#         self.loss_weight = loss_weight
#         self.box_reg_loss_type = box_reg_loss_type
#         self.smooth_l1_beta = smooth_l1_beta

#     def _subsample_labels(self, label):
#         """
#         Randomly sample a subset of positive and negative examples, and overwrite
#         the label vector to the ignore value (-1) for all elements that are not
#         included in the sample.

#         Args:
#             labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
#         """
#         pos_idx, neg_idx = subsample_labels(
#             label, self.batch_size_per_image, self.positive_fraction, 0
#         )
#         # Fill with the ignore label (-1), then set positive and negative labels
#         label.fill_(-1)
#         label.scatter_(0, pos_idx, 1)
#         label.scatter_(0, neg_idx, 0)
#         return label

    # @torch.jit.unused
    # @torch.no_grad()
    # def label_and_sample_anchors(
    #     self, anchors: List[Boxes], gt_instances: List[Instances]
    # ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    #     """
    #     Args:
    #         anchors (list[Boxes]): anchors for each feature map.
    #         gt_instances: the ground-truth instances for each image.

    #     Returns:
    #         list[Tensor]:
    #             List of #img tensors. i-th element is a vector of labels whose length is
    #             the total number of anchors across all feature maps R = sum(Hi * Wi * A).
    #             Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
    #             class; 1 = positive class.
    #         list[Tensor]:
    #             i-th element is a Rx4 tensor. The values are the matched gt boxes for each
    #             anchor. Values are undefined for those anchors not labeled as 1.
    #     """
    #     anchors = Boxes.cat(anchors)

    #     gt_boxes = [x.gt_boxes for x in gt_instances]
    #     image_sizes = [x.image_size for x in gt_instances]
    #     del gt_instances

    #     gt_labels = []
    #     matched_gt_boxes = []
    #     for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
    #         """
    #         image_size_i: (h, w) for the i-th image
    #         gt_boxes_i: ground-truth boxes for i-th image
    #         """

    #         match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
    #         matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
    #         # Matching is memory-expensive and may result in CPU tensors. But the result is small
    #         gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
    #         del match_quality_matrix

    #         if self.anchor_boundary_thresh >= 0:
    #             # Discard anchors that go out of the boundaries of the image
    #             # NOTE: This is legacy functionality that is turned off by default in Detectron2
    #             anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
    #             gt_labels_i[~anchors_inside_image] = -1

    #         # A vector of labels (-1, 0, 1) for each anchor
    #         gt_labels_i = self._subsample_labels(gt_labels_i)

    #         if len(gt_boxes_i) == 0:
    #             # These values won't be used anyway since the anchor is labeled as background
    #             matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
    #         else:
    #             # TODO wasted indexing computation for ignored boxes
    #             matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

    #         gt_labels.append(gt_labels_i)  # N,AHW
    #         matched_gt_boxes.append(matched_gt_boxes_i)
    #     return gt_labels, matched_gt_boxes

#     @torch.jit.unused
#     def losses(
#         self,
#         anchors: List[Boxes],
#         pred_objectness_logits: List[torch.Tensor],
#         gt_labels: List[torch.Tensor],
#         pred_anchor_deltas: List[torch.Tensor],
#         gt_boxes: List[torch.Tensor],
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Return the losses from a set of RPN predictions and their associated ground-truth.

#         Args:
#             anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
#                 has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
#             pred_objectness_logits (list[Tensor]): A list of L elements.
#                 Element i is a tensor of shape (N, Hi*Wi*A) representing
#                 the predicted objectness logits for all anchors.
#             gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
#             pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
#                 (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
#                 to proposals.
#             gt_boxes (list[Tensor]): Output of :meth:`label_and_sample_anchors`.

#         Returns:
#             dict[loss name -> loss value]: A dict mapping from loss name to loss value.
#                 Loss names are: `loss_rpn_cls` for objectness classification and
#                 `loss_rpn_loc` for proposal localization.
#         """
#         num_images = len(gt_labels)
#         gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))

#         # Log the number of positive/negative anchors per-image that's used in training
#         pos_mask = gt_labels == 1
#         num_pos_anchors = pos_mask.sum().item()
#         num_neg_anchors = (gt_labels == 0).sum().item()
#         storage = get_event_storage()
#         storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
#         storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

#         localization_loss = _dense_box_regression_loss(
#             anchors,
#             self.box2box_transform,
#             pred_anchor_deltas,
#             gt_boxes,
#             pos_mask,
#             box_reg_loss_type=self.box_reg_loss_type,
#             smooth_l1_beta=self.smooth_l1_beta,
#         )

#         valid_mask = gt_labels >= 0
#         objectness_loss = F.binary_cross_entropy_with_logits(
#             cat(pred_objectness_logits, dim=1)[valid_mask],
#             gt_labels[valid_mask].to(torch.float32),
#             reduction="sum",
#         )
#         normalizer = self.batch_size_per_image * num_images
#         losses = {
#             "loss_rpn_cls": objectness_loss / normalizer,
#             # The original Faster R-CNN paper uses a slightly different normalizer
#             # for loc loss. But it doesn't matter in practice
#             "loss_rpn_loc": localization_loss / normalizer,
#         }
#         losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
#         return losses

#     def forward(
#         self,
#         images: ImageList,
#         features: Dict[str, torch.Tensor],
#         gt_instances: Optional[List[Instances]] = None,
#     ):
#         """
#         Args:
#             images (ImageList): input images of length `N`
#             features (dict[str, Tensor]): input data as a mapping from feature
#                 map name to tensor. Axis 0 represents the number of images `N` in
#                 the input data; axes 1-3 are channels, height, and width, which may
#                 vary between feature maps (e.g., if a feature pyramid is used).
#             gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
#                 Each `Instances` stores ground-truth instances for the corresponding image.

#         Returns:
#             proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
#             loss: dict[Tensor] or None
#         """
#         features = [features[f] for f in self.in_features]
#         anchors = self.anchor_generator(features)

#         pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
#         # Transpose the Hi*Wi*A dimension to the middle:
#         pred_objectness_logits = [
#             # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
#             score.permute(0, 2, 3, 1).flatten(1)
#             for score in pred_objectness_logits
#         ]
#         pred_anchor_deltas = [
#             # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
#             x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
#             .permute(0, 3, 4, 1, 2)
#             .flatten(1, -2)
#             for x in pred_anchor_deltas
#         ]

#         if self.training:
#             assert gt_instances is not None, "RPN requires gt_instances in training!"
#             gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
#             losses = self.losses(
#                 anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
#             )
#         else:
#             losses = {}
#         proposals = self.predict_proposals(
#             anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
#         )
#         return proposals, losses

#     def predict_proposals(
#         self,
#         anchors: List[Boxes],
#         pred_objectness_logits: List[torch.Tensor],
#         pred_anchor_deltas: List[torch.Tensor],
#         image_sizes: List[Tuple[int, int]],
#     ):
#         """
#         Decode all the predicted box regression deltas to proposals. Find the top proposals
#         by applying NMS and removing boxes that are too small.

#         Returns:
#             proposals (list[Instances]): list of N Instances. The i-th Instances
#                 stores post_nms_topk object proposals for image i, sorted by their
#                 objectness score in descending order.
#         """
#         # The proposals are treated as fixed for joint training with roi heads.
#         # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
#         # are also network responses.
#         with torch.no_grad():
#             pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
#             return find_top_rpn_proposals(
#                 pred_proposals,
#                 pred_objectness_logits,
#                 image_sizes,
#                 self.nms_thresh,
#                 self.pre_nms_topk[self.training],
#                 self.post_nms_topk[self.training],
#                 self.min_box_size,
#                 self.training,
#             )

#     def _decode_proposals(self, anchors: List[Boxes], pred_anchor_deltas: List[torch.Tensor]):
#         """
#         Transform anchors into proposals by applying the predicted anchor deltas.

#         Returns:
#             proposals (list[Tensor]): A list of L tensors. Tensor i has shape
#                 (N, Hi*Wi*A, B)
#         """
#         N = pred_anchor_deltas[0].shape[0]
#         proposals = []
#         # For each feature map
#         for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
#             B = anchors_i.tensor.size(1)
#             pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
#             # Expand anchors to shape (N*Hi*Wi*A, B)
#             anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
#             proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
#             # Append feature map proposals with shape (N, Hi*Wi*A, B)
#             proposals.append(proposals_i.view(N, -1, B))
#         return proposals
# class Box2BoxTransform:
#     """
#     The box-to-box transform defined in R-CNN. The transformation is parameterized
#     by 4 deltas: (dx, dy, dw, dh). The transformation scales the box's width and height
#     by exp(dw), exp(dh) and shifts a box's center by the offset (dx * width, dy * height).
#     """

#     def __init__(
#         self, weights: Tuple[float, float, float, float], scale_clamp: float = math.log(1000.0 / 16)
#     ):
#         """
#         Args:
#             weights (4-element tuple): Scaling factors that are applied to the
#                 (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally set
#                 such that the deltas have unit variance; now they are treated as
#                 hyperparameters of the system.
#             scale_clamp (float): When predicting deltas, the predicted box scaling
#                 factors (dw and dh) are clamped such that they are <= scale_clamp.
#         """
#         self.weights = weights
#         self.scale_clamp = scale_clamp

#     def get_deltas(self, src_boxes, target_boxes):
#         """
#         Get box regression transformation deltas (dx, dy, dw, dh) that can be used
#         to transform the `src_boxes` into the `target_boxes`. That is, the relation
#         ``target_boxes == self.apply_deltas(deltas, src_boxes)`` is true (unless
#         any delta is too large and is clamped).

#         Args:
#             src_boxes (Tensor): source boxes, e.g., object proposals
#             target_boxes (Tensor): target of the transformation, e.g., ground-truth
#                 boxes.
#         """
#         assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
#         assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

#         src_widths = src_boxes[:, 2] - src_boxes[:, 0]
#         src_heights = src_boxes[:, 3] - src_boxes[:, 1]
#         src_ctr_x = src_boxes[:, 0] + 0.5 * src_widths
#         src_ctr_y = src_boxes[:, 1] + 0.5 * src_heights

#         target_widths = target_boxes[:, 2] - target_boxes[:, 0]
#         target_heights = target_boxes[:, 3] - target_boxes[:, 1]
#         target_ctr_x = target_boxes[:, 0] + 0.5 * target_widths
#         target_ctr_y = target_boxes[:, 1] + 0.5 * target_heights

#         wx, wy, ww, wh = self.weights
#         dx = wx * (target_ctr_x - src_ctr_x) / src_widths
#         dy = wy * (target_ctr_y - src_ctr_y) / src_heights
#         dw = ww * torch.log(target_widths / src_widths)
#         dh = wh * torch.log(target_heights / src_heights)

#         deltas = torch.stack((dx, dy, dw, dh), dim=1)
#         assert (src_widths > 0).all().item(), "Input boxes to Box2BoxTransform are not valid!"
#         return deltas

#     def apply_deltas(self, deltas, boxes):
#         """
#         Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

#         Args:
#             deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
#                 deltas[i] represents k potentially different class-specific
#                 box transformations for the single box boxes[i].
#             boxes (Tensor): boxes to transform, of shape (N, 4)
#         """
#         deltas = deltas.float()  # ensure fp32 for decoding precision
#         boxes = boxes.to(deltas.dtype)

#         widths = boxes[:, 2] - boxes[:, 0]
#         heights = boxes[:, 3] - boxes[:, 1]
#         ctr_x = boxes[:, 0] + 0.5 * widths
#         ctr_y = boxes[:, 1] + 0.5 * heights

#         wx, wy, ww, wh = self.weights
#         dx = deltas[:, 0::4] / wx
#         dy = deltas[:, 1::4] / wy
#         dw = deltas[:, 2::4] / ww
#         dh = deltas[:, 3::4] / wh

#         # Prevent sending too large values into torch.exp()
#         dw = torch.clamp(dw, max=self.scale_clamp)
#         dh = torch.clamp(dh, max=self.scale_clamp)

#         pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
#         pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
#         pred_w = torch.exp(dw) * widths[:, None]
#         pred_h = torch.exp(dh) * heights[:, None]

#         x1 = pred_ctr_x - 0.5 * pred_w
#         y1 = pred_ctr_y - 0.5 * pred_h
#         x2 = pred_ctr_x + 0.5 * pred_w
#         y2 = pred_ctr_y + 0.5 * pred_h
#         pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
#         return pred_boxes.reshape(deltas.shape)

# class Matcher:
#     """
#     This class assigns to each predicted "element" (e.g., a box) a ground-truth
#     element. Each predicted element will have exactly zero or one matches; each
#     ground-truth element may be matched to zero or more predicted elements.

#     The matching is determined by the MxN match_quality_matrix, that characterizes
#     how well each (ground-truth, prediction)-pair match each other. For example,
#     if the elements are boxes, this matrix may contain box intersection-over-union
#     overlap values.

#     The matcher returns (a) a vector of length N containing the index of the
#     ground-truth element m in [0, M) that matches to prediction n in [0, N).
#     (b) a vector of length N containing the labels for each prediction.
#     """

#     def __init__(
#         self, thresholds: List[float], labels: List[int], allow_low_quality_matches: bool = False
#     ):
#         """
#         Args:
#             thresholds (list): a list of thresholds used to stratify predictions
#                 into levels.
#             labels (list): a list of values to label predictions belonging at
#                 each level. A label can be one of {-1, 0, 1} signifying
#                 {ignore, negative class, positive class}, respectively.
#             allow_low_quality_matches (bool): if True, produce additional matches
#                 for predictions with maximum match quality lower than high_threshold.
#                 See set_low_quality_matches_ for more details.

#             For example,
#                 thresholds = [0.3, 0.5]
#                 labels = [0, -1, 1]
#                 All predictions with iou < 0.3 will be marked with 0 and
#                 thus will be considered as false positives while training.
#                 All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
#                 thus will be ignored.
#                 All predictions with 0.5 <= iou will be marked with 1 and
#                 thus will be considered as true positives.
#         """
#         # Add -inf and +inf to first and last position in thresholds
#         thresholds = thresholds[:]
#         assert thresholds[0] > 0
#         thresholds.insert(0, -float("inf"))
#         thresholds.append(float("inf"))
#         # Currently torchscript does not support all + generator
#         assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])])
#         assert all([l in [-1, 0, 1] for l in labels])
#         assert len(labels) == len(thresholds) - 1
#         self.thresholds = thresholds
#         self.labels = labels
#         self.allow_low_quality_matches = allow_low_quality_matches

#     def __call__(self, match_quality_matrix):
#         """
#         Args:
#             match_quality_matrix (Tensor[float]): an MxN tensor, containing the
#                 pairwise quality between M ground-truth elements and N predicted
#                 elements. All elements must be >= 0 (due to the us of `torch.nonzero`
#                 for selecting indices in :meth:`set_low_quality_matches_`).

#         Returns:
#             matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
#                 ground-truth index in [0, M)
#             match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
#                 whether a prediction is a true or false positive or ignored
#         """
#         assert match_quality_matrix.dim() == 2
#         if match_quality_matrix.numel() == 0:
#             default_matches = match_quality_matrix.new_full(
#                 (match_quality_matrix.size(1),), 0, dtype=torch.int64
#             )
#             # When no gt boxes exist, we define IOU = 0 and therefore set labels
#             # to `self.labels[0]`, which usually defaults to background class 0
#             # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
#             default_match_labels = match_quality_matrix.new_full(
#                 (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
#             )
#             return default_matches, default_match_labels

#         assert torch.all(match_quality_matrix >= 0)

#         # match_quality_matrix is M (gt) x N (predicted)
#         # Max over gt elements (dim 0) to find best gt candidate for each prediction
#         matched_vals, matches = match_quality_matrix.max(dim=0)

#         match_labels = matches.new_full(matches.size(), 1, dtype=torch.int8)

#         for l, low, high in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
#             low_high = (matched_vals >= low) & (matched_vals < high)
#             match_labels[low_high] = l

#         if self.allow_low_quality_matches:
#             self.set_low_quality_matches_(match_labels, match_quality_matrix)

#         return matches, match_labels

#     def set_low_quality_matches_(self, match_labels, match_quality_matrix):
#         """
#         Produce additional matches for predictions that have only low-quality matches.
#         Specifically, for each ground-truth G find the set of predictions that have
#         maximum overlap with it (including ties); for each prediction in that set, if
#         it is unmatched, then match it to the ground-truth G.

#         This function implements the RPN assignment case (i) in Sec. 3.1.2 of
#         :paper:`Faster R-CNN`.
#         """
#         # For each gt, find the prediction with which it has highest quality
#         highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
#         # Find the highest quality match available, even if it is low, including ties.
#         # Note that the matches qualities must be positive due to the use of
#         # `torch.nonzero`.
#         _, pred_inds_with_highest_quality = nonzero_tuple(
#             match_quality_matrix == highest_quality_foreach_gt[:, None]
#         )
#         # If an anchor was labeled positive only due to a low-quality match
#         # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
#         # This follows the implementation in Detectron, and is found to have no significant impact.
#         match_labels[pred_inds_with_highest_quality] = 1

# class DefaultAnchorGenerator(nn.Module):
#     """
#     Compute anchors in the standard ways described in
#     "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks".
#     """

#     box_dim: torch.jit.Final[int] = 4
#     """
#     the dimension of each anchor box.
#     """

#     @configurable
#     def __init__(self, *, sizes, aspect_ratios, strides, offset=0.5):
#         """
#         This interface is experimental.

#         Args:
#             sizes (list[list[float]] or list[float]):
#                 If ``sizes`` is list[list[float]], ``sizes[i]`` is the list of anchor sizes
#                 (i.e. sqrt of anchor area) to use for the i-th feature map.
#                 If ``sizes`` is list[float], ``sizes`` is used for all feature maps.
#                 Anchor sizes are given in absolute lengths in units of
#                 the input image; they do not dynamically scale if the input image size changes.
#             aspect_ratios (list[list[float]] or list[float]): list of aspect ratios
#                 (i.e. height / width) to use for anchors. Same "broadcast" rule for `sizes` applies.
#             strides (list[int]): stride of each input feature.
#             offset (float): Relative offset between the center of the first anchor and the top-left
#                 corner of the image. Value has to be in [0, 1).
#                 Recommend to use 0.5, which means half stride.
#         """
#         super().__init__()

#         self.strides = strides
#         self.num_features = len(self.strides)
#         sizes = _broadcast_params(sizes, self.num_features, "sizes")
#         aspect_ratios = _broadcast_params(aspect_ratios, self.num_features, "aspect_ratios")
#         self.cell_anchors = self._calculate_anchors(sizes, aspect_ratios)

#         self.offset = offset
#         assert 0.0 <= self.offset < 1.0, self.offset

#     # @classmethod
#     # def from_config(cls, cfg, input_shape: List[ShapeSpec]):
#     #     return {
#     #         "sizes": cfg.MODEL.ANCHOR_GENERATOR.SIZES,
#     #         "aspect_ratios": cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
#     #         "strides": [x.stride for x in input_shape],
#     #         "offset": cfg.MODEL.ANCHOR_GENERATOR.OFFSET,
#     #     }

#     def _calculate_anchors(self, sizes, aspect_ratios):
#         cell_anchors = [
#             self.generate_cell_anchors(s, a).float() for s, a in zip(sizes, aspect_ratios)
#         ]
#         return BufferList(cell_anchors)

#     @torch.jit.unused
#     def num_cell_anchors(self):
#         """
#         Alias of `num_anchors`.
#         """
#         return self.num_anchors

#     @torch.jit.unused
#     def num_anchors(self):
#         """
#         Returns:
#             list[int]: Each int is the number of anchors at every pixel
#                 location, on that feature map.
#                 For example, if at every pixel we use anchors of 3 aspect
#                 ratios and 5 sizes, the number of anchors is 15.
#                 (See also ANCHOR_GENERATOR.SIZES and ANCHOR_GENERATOR.ASPECT_RATIOS in config)

#                 In standard RPN models, `num_anchors` on every feature map is the same.
#         """
#         return [len(cell_anchors) for cell_anchors in self.cell_anchors]

#     def _grid_anchors(self, grid_sizes: List[List[int]]):
#         """
#         Returns:
#             list[Tensor]: #featuremap tensors, each is (#locations x #cell_anchors) x 4
#         """
#         anchors = []
#         # buffers() not supported by torchscript. use named_buffers() instead
#         buffers: List[torch.Tensor] = [x[1] for x in self.cell_anchors.named_buffers()]
#         for size, stride, base_anchors in zip(grid_sizes, self.strides, buffers):
#             shift_x, shift_y = _create_grid_offsets(size, stride, self.offset, base_anchors)
#             shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

#             anchors.append((shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4))

#         return anchors

#     def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
#         """
#         Generate a tensor storing canonical anchor boxes, which are all anchor
#         boxes of different sizes and aspect_ratios centered at (0, 0).
#         We can later build the set of anchors for a full feature map by
#         shifting and tiling these tensors (see `meth:_grid_anchors`).

#         Args:
#             sizes (tuple[float]):
#             aspect_ratios (tuple[float]]):

#         Returns:
#             Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
#                 in XYXY format.
#         """

#         # This is different from the anchor generator defined in the original Faster R-CNN
#         # code or Detectron. They yield the same AP, however the old version defines cell
#         # anchors in a less natural way with a shift relative to the feature grid and
#         # quantization that results in slightly different sizes for different aspect ratios.
#         # See also https://github.com/facebookresearch/Detectron/issues/227

#         anchors = []
#         for size in sizes:
#             area = size**2.0
#             for aspect_ratio in aspect_ratios:
#                 # s * s = w * h
#                 # a = h / w
#                 # ... some algebra ...
#                 # w = sqrt(s * s / a)
#                 # h = a * w
#                 w = math.sqrt(area / aspect_ratio)
#                 h = aspect_ratio * w
#                 x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
#                 anchors.append([x0, y0, x1, y1])
#         return torch.tensor(anchors)

#     def forward(self, features: List[torch.Tensor]):
#         """
#         Args:
#             features (list[Tensor]): list of backbone feature maps on which to generate anchors.

#         Returns:
#             list[Boxes]: a list of Boxes containing all the anchors for each feature map
#                 (i.e. the cell anchors repeated over all locations in the feature map).
#                 The number of anchors of each feature map is Hi x Wi x num_cell_anchors,
#                 where Hi, Wi are resolution of the feature map divided by anchor stride.
#         """
#         grid_sizes = [feature_map.shape[-2:] for feature_map in features]
#         anchors_over_all_feature_maps = self._grid_anchors(grid_sizes)  # pyre-ignore
#         return [x for x in anchors_over_all_feature_maps]

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def _create_grid_offsets(
    size: List[int], stride: int, offset: float, target_device_tensor: torch.Tensor
):
    grid_height, grid_width = size
    shifts_x = move_device_like(
        torch.arange(offset * stride, grid_width * stride, step=stride, dtype=torch.float32),
        target_device_tensor,
    )
    shifts_y = move_device_like(
        torch.arange(offset * stride, grid_height * stride, step=stride, dtype=torch.float32),
        target_device_tensor,
    )

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


def _broadcast_params(params, num_features, name):
    """
    If one size (or aspect ratio) is specified and there are multiple feature
    maps, we "broadcast" anchors of that single size (or aspect ratio)
    over all feature maps.

    If params is list[float], or list[list[float]] with len(params) == 1, repeat
    it num_features time.

    Returns:
        list[list[float]]: param for each feature
    """
    assert isinstance(
        params, collections.abc.Sequence
    ), f"{name} in anchor generator has to be a list! Got {params}."
    assert len(params), f"{name} in anchor generator cannot be empty!"
    if not isinstance(params[0], collections.abc.Sequence):  # params is list[float]
        return [params] * num_features
    if len(params) == 1:
        return list(params) * num_features
    assert len(params) == num_features, (
        f"Got {name} of length {len(params)} in anchor generator, "
        f"but the number of input features is {num_features}!"
    )
    return params

def nonzero_tuple(x):
    """
    A 'as_tuple=True' version of torch.nonzero to support torchscript.
    because of https://github.com/pytorch/pytorch/issues/38718
    """
    if torch.jit.is_scripting():
        if x.dim() == 0:
            return x.unsqueeze(0).nonzero().unbind(1)
        return x.nonzero().unbind(1)
    else:
        return x.nonzero(as_tuple=True)