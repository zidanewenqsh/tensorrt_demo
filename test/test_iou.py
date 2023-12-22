import torch
import torchvision.ops as ops

# 定义两组边界框
boxes1 = torch.tensor([[0, 0, 2, 2],
                       [1, 1, 3, 3]], dtype=torch.float32)
boxes2 = torch.tensor([[1, 1, 3, 3],
                       [0, 0, 2, 2]], dtype=torch.float32)

# 计算IoU
ious = ops.box_iou(boxes1, boxes2)

print(ious) # 定义一组边界框和对应的置信度分数

boxes = torch.tensor([[10, 10, 20, 20],
                      [15, 15, 25, 25],
                      [20, 20, 30, 30]], dtype=torch.float32)
scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

# 定义IoU阈值
iou_threshold = 0.5

# 应用NMS
keep = ops.nms(boxes, scores, iou_threshold)

print(keep)