from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
training_data = datasets.MNIST(root='./myData',train=True,transform=ToTensor(),download=True)
test_data = datasets.MNIST(root='./myData',train=False,transform=ToTensor(),download=True)
train_dataloader = DataLoader(training_data,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size=64,shuffle=True)
#figure=plt.figure()
#img,label=training_data[10]
#plt.title(label)
#plt.imshow(img.squeeze(),cmap='gray')
#plt.show()
#print(img.shape)
class lstm_net(nn.Module):
    def __init__(self):
        super(lstm_net,self).__init__()
        self.lstm = nn.LSTM(input_size=28,hidden_size=64,num_layers=1,batch_first=True)
        self.linear=nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,10)
        )
    def forward(self,x):
        r_out,(h_n,h_c)=self.lstm(x,None) # None means zero initial hidden state
        out=self.linear(r_out[:,-1,:])
        return out
rnn=lstm_net().cuda()
print(rnn)
optimizer=torch.optim.Adam(rnn.parameters(),lr=0.01)
loss_func=nn.CrossEntropyLoss()
epochs=100
for epoch in range(epochs):
    rnn.train()
    for step,(x,y)in enumerate(train_dataloader):
        x=x.view(-1,28,28).cuda()
        y=y.cuda()
        output=rnn(x)
        loss=loss_func(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%100==0:
            print('Epoch:',epoch,'|train loss:%.4f'%loss.item())
    rnn.eval()
    with torch.no_grad():
        test_loss=0
        correct=0
        num_batches=len(test_dataloader)
        size=len(test_dataloader.dataset)
        for x,y in test_dataloader:
            x=x.view(-1,28,28).cuda()
            y=y.cuda()
            output=rnn(x)
            test_loss+=loss_func(output,y).item()
            pred=torch.argmax(output,dim=1)
            correct+=torch.eq(pred,y).sum().item()
        test_loss/=num_batches
        acc=correct/size
        print('Test loss:%.4f'%test_loss,'|accuracy:%.2f%%'%(100*acc))
        
