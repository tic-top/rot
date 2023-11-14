from model import regnet
from dataset import MyDataset
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # paser num of workers
    parser.add_argument(
        "--workers", "-w", type=int, default=15
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default="./data"
    )
    opts = parser.parse_args()
    
    input_size = 224
    traindir = opts.data_dir
    batch_size = 64
    report = 100
    me = 30
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    whole_dataset = MyDataset(traindir)
    # Define the size for your train and test data
    train_size = int(0.8 * len(whole_dataset))
    test_size = len(whole_dataset) - train_size
    # Split the dataset
    train_dataset, test_dataset = random_split(whole_dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=opts.workers, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=opts.workers, pin_memory=True, prefetch_factor=2)
    net = regnet(n_class=180, train=True)
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=me)
    best_acc = 0
    start_time = time.time()
    for epoch in range(me):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True) 
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() 
            if i % report == report - 1:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/report))
                running_loss = 0.0
        scheduler.step()
        # test
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device) 
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
        print('Accuracy on epoch %d: %d %%' % (epoch+1, 100*correct/total))
        print("time per epoch %d: %d s" % (epoch+1, (time.time()-start_time)/(epoch+1)))
        
        # save model
        if int(100*correct/total) > best_acc:
            best_acc = int(100*correct/total)
            torch.save(net.state_dict(), f'regnet_best.pth')
        elif epoch % 10 ==9:
            torch.save(net.state_dict(), f'regnet_epoch{epoch+1}_acc{int(100*correct/total)}.pth')

    print('Finished Training')
