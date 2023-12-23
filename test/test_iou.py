import torch
import torchvision.ops as ops

import re
import numpy as np
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


# 定义一个解析行的函数
def parse_line(line):
    pattern = r"x1: (\d+\.\d+), y1: (\d+\.\d+), x2: (\d+\.\d+), y2: (\d+\.\d+), prob: (\d+\.\d+), label: (\d+), count: (\d+)"
    match = re.match(pattern, line)
    
    if match:
        return {
            'x1': float(match.group(1)),
            'y1': float(match.group(2)),
            'x2': float(match.group(3)),
            'y2': float(match.group(4)),
            'prob': float(match.group(5)),
            'label': int(match.group(6)),
            'count': int(match.group(7))
        }
    else:
        return None

# 定义一个解析行的函数
def parse_line2(line):
    pattern = r"x1: (\d+\.\d+), y1: (\d+\.\d+), x2: (\d+\.\d+), y2: (\d+\.\d+), prob: (\d+\.\d+), label: (\d+), count: (\d+)"
    match = re.match(pattern, line)
    
    if match:
        return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5)), int(match.group(6)), int(match.group(7))]
    else:
        return None

if __name__ == '__main__':
    with open('test/data.txt', 'r') as file:
        lines = file.readlines()

    # 解析每一行数据
    data = [parse_line(line) for line in lines if line.strip()]

    # 打印解析后的数据
    # for item in data:
    #     print(item)
    # for k, v in data.items():
    #     print(k, v)
    np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
    print(data)
    # 解析每一行数据并组成二维数组
    data1 = [parse_line2(line) for line in lines if line.strip()]
    data1 = np.array(data1)

    # 打印二维数组
    print(data1.shape)
    data1_0 = data1[data1[:, 5] == 0]
    print(data1_0)
    torch.save(data1_0, 'test/data1_0.pt')
    data1_0 = torch.load('test/data1_0.pt')
    coord = torch.from_numpy(data1_0[:, :4])
    # print(coord)
    # print(coord.shape)
    probs = torch.from_numpy(data1_0[:, 4])
    # print(probs)
    # print(probs.shape)
    keep = ops.nms(coord, probs, 0.5)
    print(keep)
    print(data1_0[keep])
# [[212.6 239.9 285.2 513.4   0.9   0.   16. ]
#  [476.4 232.4 559.1 523.3   0.8   0.    5. ]
#  [110.  235.2 224.1 535.6   0.8   0.    3. ]
#  [ 80.4 326.8 128.  528.2   0.4   0.   32. ]]