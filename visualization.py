
import numpy as np

from metrics import *

def visualization(x, mask, color=255, weight=0):
    '''
    visualization for semantic segmentation
    :param x: pytorch tensor [1, 3, m, n]    0<=x<=1 so you need to * 255 to visualize it
    :param mask: pytorch tensor [1,1, m, n],   values are 0 or 1, is the mask.
    :return: a PIL image which can directly image.show
    '''

    mask_tensor = mask * color / 255
    non_masked = (1 - mask) * x  # 没mask的地方有数值
    masked = mask * x * weight + mask_tensor * mask * (1 - weight)  # mask的地方有数值
    x = non_masked + masked
    x = x.permute(0, 2, 3, 1)
    x = x.squeeze().numpy()
    x *= 255
    x = np.uint8(x)

    x = Image.fromarray(x)
    x.show()


def visualize_one(x, mask = None, threshold=0.834):
    if mask is None:
        mask = torch.zeros_like(x)
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        mask = mask.unsqueeze(0)
    if x.shape[0] != 1:
        x = x[0].unsqueeze(0)
        mask = mask[0].unsqueeze(0)
    y = mask>threshold
    y = y.to(torch.float)
    visualization(x, y)

if __name__ == '__main__':
    from unet_model import UNet
    import torchvision
    model = AttU_Net(4,1)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cuda')))
    model.eval()

    from data import get_loader

    loader=get_loader(1,"./WholeData/valid/image/","./WholeData/valid/label/",Type="test")

    with torch.no_grad():
        for x, y in loader:
            pre = model(x)
            pre=torch.sigmoid(pre)
            pre= Binaryzation(pre, 0.93)


            visualize_one(x[:,0:3,], pre)#可视化预测结果 前三层是原图，第四层是LoG图
            visualize_one(x[:,0:3,], y)#可视化truth label

