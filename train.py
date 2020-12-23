import numpy as np
import math
import random
import os
import torch
import torch.nn.functional as F
from path import Path
from source import model as m
from source import dataset
from source import utils
from source.args import parse_args
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

random.seed = 42

args = parse_args()
path = Path(args.root_dir)

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)};

train_transforms = transforms.Compose([
    utils.PointSampler(1024),
    utils.Normalize(),
    utils.RandRotation_z(),
    utils.RandomNoise(),
    utils.ToTensor()
])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

 
train_dataset = dataset.PointCloudData(path, transform=train_transforms)

test_dataset = dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size*2)

def train(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 16 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())
            
def evaluate(model, data_loader, loss_history):
    model.eval()
    
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')

N_EPOCHS = args.epochs
model = m.ImageTransformer()
model = model.double()
#model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)


train_loss_history, test_loss_history = [], []
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    #start_time = time.time()
    train(model, optimizer, train_loader, train_loss_history)
    #print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    #evaluate(model, test_loader, test_loss_history)

print('Execution time')


model.eval()
correct = total = 0

# validation
if test_loader:
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            inputs, labels = data, target
            outputs, __, __ = model(inputs.transpose(1,2))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100. * correct / total
    print('Valid accuracy: %d %%' % val_acc)
        # save the model
        
        
PATH = ".\ViTnet_Cifar10_4x4_aug_1.pt" # Use your own path
torch.save(model.state_dict(), PATH)