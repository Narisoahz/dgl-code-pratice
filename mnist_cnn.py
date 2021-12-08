import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cuda" if torch.cuda.is_available() else "cpu"

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3),  #(b,16,26,26)因为是黑白图片，只有一个通道，所以输入深度为1；输出深度为16，表示有16个神经元
            nn.BatchNorm2d(16), #规范化
            nn.ReLU(inplace=True))
    
        self.layer2 = nn.Sequential( 
            nn.Conv2d(16,32,kernel_size=3), #(b,32,24,24)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))  #(b,32,12,12)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),  #(b,64,10,10)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3),  #(b,128,8,8)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2))  #(b,128,4,4)
        
        self.fc = nn.Sequential(
            nn.Linear(128*4*4,1024),  
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10))
        
    def forward(self,x):
        x = self.layer1(x)  #数据(b,1,28,28),卷积核(1,16,3,3)——(b,16,26,26)  
        x = self.layer2(x)  #(b,16,26,26) ,(16,32,3,3) ——（b,32,24,24）——（b,32,12,12）
        x = self.layer3(x)  #(b,32,12,12）,(32,64,3,3)) ——(b,10,10,64)
        x = self.layer4(x)  #(b,10,10,64),(64,128,3,3))——(b,128,8,8)——(b,128,4,4)
        x = x.view(x.size(0),-1) #(b,128,4,4)——(b,128*4*4)
        x = self.fc(x)  #(b,128*4*4)*(128*4*4,1024)——(b,1024)——(b,1024)*(1024,128)——(b,128)——(b,128)*(128,10)——(b,10)
        return x

model=CNN().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

#训练模型
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")