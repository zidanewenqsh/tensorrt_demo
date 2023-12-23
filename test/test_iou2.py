class Box:
    def __init__(self, x1, y1, x2, y2, prob, label):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.prob = prob
        self.label = label
        self.remove = False

    def __repr__(self):
        return f"Box(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, prob={self.prob}, label={self.label}, remove={self.remove})"
import torch

def iou(box1, box2):
    """计算两个边界框的交并比（IOU）"""
    xx1 = max(box1.x1, box2.x1)
    yy1 = max(box1.y1, box2.y1)
    xx2 = min(box1.x2, box2.x2)
    yy2 = min(box1.y2, box2.y2)

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h

    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

def nms(boxes, nms_threshold):
    """执行非极大值抑制"""
    if len(boxes) == 0:
        return []

    # 按置信度排序
    boxes = sorted(boxes, key=lambda x: x.prob, reverse=True)

    for i in range(len(boxes)):
        if boxes[i].remove:
            continue
        for j in range(i + 1, len(boxes)):
            if boxes[j].remove:
                continue
            if iou(boxes[i], boxes[j]) > nms_threshold:
                if boxes[i].prob > boxes[j].prob:
                    boxes[j].remove = True
                else:
                    boxes[i].remove = True
                    break

    return [box for box in boxes if not box.remove]

# 示例使用
boxes = [
    Box(100, 100, 200, 200, 0.9, 1),
    Box(150, 150, 250, 250, 0.8, 1),
    Box(300, 300, 400, 400, 0.7, 2)
]

filtered_boxes = nms(boxes, nms_threshold=0.5)
for box in filtered_boxes:
    print(box)

data1_0 = torch.load('test/data1_0.pt')
print(data1_0.shape)
boxes = []
for i in range(data1_0.shape[0]):
    boxes.append(Box(data1_0[i, 0], data1_0[i, 1], data1_0[i, 2], data1_0[i, 3], data1_0[i, 4], data1_0[i, 5]))

filtered_boxes = nms(boxes, nms_threshold=0.5)
for box in filtered_boxes:
    print(box)