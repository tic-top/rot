from model import mobilenetv3
from dataset import MyDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

traindir = "./google-street-view"
whole_dataset = MyDataset(traindir)
# Define the size for your train and test data
train_size = int(0.8 * len(whole_dataset))
test_size = len(whole_dataset) - train_size
train_dataset, test_dataset = random_split(whole_dataset, [train_size, test_size])
val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=15, pin_memory=True, prefetch_factor=2)
mode = "large"

net = mobilenetv3(n_class=180, input_size = 224, mode = mode)
net.load_state_dict(torch.load(f'mobilenetv3_{mode}.pth'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
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
        # print(predicted)
        # print(labels)
print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct/total))

