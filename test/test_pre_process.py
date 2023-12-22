import cv2
import numpy as np
def letterbox1(src, w, h):
    height, width = src.shape[:2]

    scale = min(float(w) / width, float(h) / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized = cv2.resize(src, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 创建输出图像并填充灰色背景
    out = np.full((h, w, 3), 114, dtype=np.uint8)

    # 计算放置缩放图像的位置
    x_offset = (w - new_width) // 2
    y_offset = (h - new_height) // 2

    # 将缩放后的图像放置在输出图像的中心
    out[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized

    return out
def letterbox(src, w, h):
    height, width = src.shape[:2]

    # 计算缩放比例
    scale = min(float(w) / width, float(h) / height)

    # 计算缩放后的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 计算仿射变换矩阵
    i2d = np.array([[scale, 0, (-width * scale + w) / 2],
                    [0, scale, (-height * scale + h) / 2]], dtype=np.float32)

    # 创建输出图像并填充灰色背景
    out = np.full((h, w, 3), 114, dtype=np.uint8)

    # 应用仿射变换
    out = cv2.warpAffine(src, i2d, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    return out
def preprocess(img, w, h, isX=False):
    # 如果图像是BGR格式，则转换为RGB
    processed = img.copy()
    processed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # 应用letterbox变换
    processed = letterbox(processed, w, h)

    # 转换为float32并归一化
    processed = processed.astype(np.float32) / 255.0

    # 标准化图像
    if isX:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed = (processed - mean) / std

    # 调整通道顺序为 (c, h, w)
    ret = np.zeros((3 * w * h), dtype=np.float32)
    cs = w * h
    # 按通道顺序存储每个通道的所有像素值
    for c in range(3):
        for i in range(h):
            for j in range(w):
                index = i * w + j
                ret[cs * c + index] = processed[i, j, c]
                # ret[c, index] = processed[i, j, c]
    ret = ret.reshape(3, h, w)
    processed = np.transpose(processed, (2, 0, 1))
    return processed, ret

# 示例使用
if __name__ == '__main__':
    c = 3
    w = 4
    h = 4
    data = np.arange(c * w * h, dtype=np.uint8).reshape(h, w, c)
    print(data)
    # data2 = preprocess(data, 2, 2, False)
    data2, data3 = preprocess(data, 2, 2, True)
    print(data2)
    print(data3)
    # data3 = letterbox(data, 2, 2)
    # print(data3)
    # img = cv2.imread('test/cat01.jpg')
    # img2 = letterbox(img, 640, 640);
    # cv2.imwrite("cat01_letterbox.jpg", img2);


# [[[0.00784314 0.03137255]
#   [0.10196079 0.1254902 ]]

#  [[0.00392157 0.02745098]
#   [0.09803922 0.12156863]]

#  [[0.         0.02352941]
#   [0.09411765 0.11764706]]]

# [[[10 16]
#   [34 40]]

#  [[ 9 15]
#   [33 39]]

#  [[ 8 14]
#   [32 38]]]


# [[[-2.08365442 -1.98090589]
#   [-1.67266032 -1.56991178]]

#  [[-2.01820728 -1.91316527]
#   [-1.59803921 -1.4929972 ]]

#  [[-1.80444444 -1.69986928]
#   [-1.38614378 -1.28156863]]]