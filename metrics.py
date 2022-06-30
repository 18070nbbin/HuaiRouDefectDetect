
def PassRate(pre, y):
    return torch.sum((y == 1)*(pre == 0))/torch.sum(y)
from dataUtils import *
def Precision(pre, y):
    TP=torch.sum((y == 1)*(pre == 1))
    return TP/torch.sum(pre)
def Recall(pre,y):
    TP=torch.sum((y == 1)*(pre == 1))
    return TP/torch.sum(y)
def IOU(pre, y):
    intersec=(y==1)*(pre==1)
    union= ((y==1) + (pre==1))>0
    return torch.sum(intersec)/torch.sum(union)

def Binaryzation(pre,threshold=0.834):

    mask=pre>=threshold
    pre[mask]=1
    mask=pre<threshold
    pre[mask]=0
    return pre

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    from Model_Zoo import *

    model =AttU_Net(4,1)
    model.to(device)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    from data import get_loader
    loader = get_loader(1, "./WholeData/valid/image/", "./WholeData/valid/label/",Type="test")#测试时，输入整张图片
    iou_list=[]
    rec_list=[]
    precision_list=[]
    passRate_list=[]
    for x,y in loader:
        x=x.to(device)
        y=y.to(device)
        pre = model(x)
        pre = torch.sigmoid(pre)
        pre = Binaryzation(pre, 0.8337)

        iou = IOU(pre, y).item()
        precision = Precision(pre, y).item()
        recall = Recall(pre, y).item()
        pass_rate=PassRate(pre,y).item()
        print("iou: ",iou,"precision: ",precision,"recall: ",recall,"pass_rate: ",pass_rate)
        passRate_list.append(pass_rate)
        iou_list.append(iou)
        precision_list.append(precision)
        rec_list.append(recall)
    print('iou:',sum(iou_list)/len(iou_list))
    print('passRate:',sum(passRate_list)/len(passRate_list))
    print('recall:', sum(rec_list) / len(rec_list))
    print('precision:', sum(precision_list) / len(precision_list))
    print('avg:',(sum(rec_list) / len(rec_list)+sum(precision_list) / len(precision_list))*0.5)

    rows = zip(iou_list, precision_list, rec_list, passRate_list)
    import csv
    with open('./test.csv', "w") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
