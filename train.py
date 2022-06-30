import torch
import torch.nn as nn
import torchvision.models.segmentation
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import get_loader
from torch.cuda.amp import autocast, GradScaler
import os
from Model_Zoo import*
def train(model: nn.Module,
          loader: DataLoader,
          valid_loader: DataLoader,
          lr=1e-4,
          weight_decay=1e-5,
          total_epoch=99,
          is_fp16=False,
          ):
    if os.path.exists('./model.pth'):
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cuda')))
    scaler = GradScaler()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer=torch.optim.SGD(model.parameters(),nesterov=True,momentum=0.9, lr=lr,weight_decay=weight_decay)

    from criterion import focal_loss
    criterion = focal_loss
    for epoch in range(1, total_epoch + 1):
        model.train()
        pbar = tqdm(loader)
        train_loss = 0
        train_acc = 0
        for step, (x, y) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)
            if is_fp16:
                with autocast():
                    x = model(x)
                    loss = criterion(x, y)
            else:
                x = model(x)
                loss = criterion(x, y)

            train_loss += loss.item()
            optimizer.zero_grad()
            if is_fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if step % 10 == 0:
                pbar.set_postfix_str(f'loss = {train_loss/(step+1)}')

        print(f'epoch {epoch}, train loss = {train_loss/len(pbar)}')

        with torch.no_grad():
            model.eval()
            pbar = tqdm(valid_loader)
            valid_loss = 0
            valid_acc = 0
            for step, (x, y) in enumerate(pbar):
                x = x.to(device)
                y = y.to(device)
                x = model(x)
                loss = criterion(x, y)
                valid_loss += loss.item()  #

        print(f'epoch {epoch}, valid loss = {valid_loss/(len(pbar))}')
        torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':


    import argparse

    paser = argparse.ArgumentParser()
    paser.add_argument('-b', '--batch_size', default=10)
    paser.add_argument('-t', '--total_epoch', default=50)
    paser.add_argument('-l', '--lr', default=1e-4)
    paser.add_argument('-w', '--weight_decay', default=1e-5)
    args = paser.parse_args()
    batch_size = int(args.batch_size)
    total_epoch = int(args.total_epoch)
    weight_decay=float(args.weight_decay)
    lr = float(args.lr)

    train_loader=get_loader(batch_size,"./CropImage/train/image/","./CropImage/train/label/")
    valid_loader=get_loader(batch_size,"./WholeData/valid/image/","./WholeData/valid/label/",Type="valid")
    model = AttU_Net(4,1)

    train(model,train_loader,valid_loader,lr=lr,
          weight_decay=weight_decay,total_epoch=total_epoch,is_fp16=True)
