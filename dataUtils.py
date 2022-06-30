"""添加各种数据增强算法"""

import cv2
import torch
from PIL import Image
from torchvision.transforms import ToPILImage
def LOG_pro(path="./")->Image:
    img = cv2.imread(path)    # cv2.imread()接口读图像，读进来直接是BGR 格式数据格式在 0~255，通道格式为(H,W,C)
    KIKI_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.cvtColor() 颜色空间转换函数。 cv2.COLOR_BGR2RGB 将BGR转为RGB颜色空间
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # cv2.cvtColor() 颜色空间转换函数。 cv2.COLOR_BGR2GRAY 将BGR转为灰色颜色空间
    gaussian = cv2.GaussianBlur(grayImage, (3,3), 0)
    dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)
    LOG = cv2.convertScaleAbs(dst)
    LOG=torch.from_numpy(LOG)
    LOG=torch.unsqueeze(LOG,0)
    return ToPILImage()(LOG)