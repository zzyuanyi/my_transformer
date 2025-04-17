from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
training_data = datasets.MNIST(root='./myData',train=True,transform=ToTensor(),download=True)
test_data = datasets.MNIST(root='./myData',train=False,transform=ToTensor(),download=True)
train_dataloader = DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoderConv = nn.Sequential(
            nn.Conv2d(1,16,3,stride=3,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16,8,3,stride=1,padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.decoderConv = nn.Sequential(
            nn.ConvTranspose2d(8,16,3,stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16,8,5,stride=3,padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8,1,2,stride=2,padding=1),
            nn.ReLU(True)
        )
    def forward(self,x):
        x1=self.encoderConv(x)
        x2=self.decoderConv(x1)
        return x1,x2
model=AutoEncoder().cuda()
print(model)
loss_fn=nn.MSELoss()
learning_rate=0.001
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,
weight_decay=1e-5)
epochs_num=100
testdataGot=0
for epoch in range(epochs_num):
    model.train()
    for i,data in enumerate(train_dataloader):
        output1,output=model(data[0].cuda())
        loss=loss_fn(output,data[0].cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i==5 and testdataGot==0:
            testdataGot=1
            testdata=data[0].clone().cuda()
            print(testdata.shape)
    print('Epoch:',epoch,'|train loss:%.4f'%loss.item())
    model.eval()
    if epoch%10==0:
        with torch.no_grad():
            figure=plt.figure()
            outp2,outp=model(testdata)
            outp=outp.cpu()
            for i in range(8):
                for j in range(4):
                    adisplay=outp[i+j*8]
                    adisplay=adisplay.squeeze().detach().numpy()
                    plt.subplot(4,8,i+j*8+1)
                    plt.imshow(adisplay,cmap='gray')
            plt.show()
def add_noise(image,prob):
    noise_out=np.zeros(image.shape.np.float)
    thres=1-prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rand=np.random.rand()
            if rand<prob:
                noise_out[i][j]=random.random()
            elif rand>thres:
                noise_out[i][j]=random.random()
            else:
                noise_out[i][j]=image[i][j]
    return noise_out
transl=ToTensor()
img,label=test_data[0]
img=img.squeeze().numpy()
print(img.shape)
img2=sp_noise(img,0.05)
