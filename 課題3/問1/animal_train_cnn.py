from sklearn.model_selection import train_test_split
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import copy
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


'''
画像に映る動物の種類をcnnで21クラスで分類する
'''

# Transform を作成する。
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomAffine([0,30], scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
# Dataset を作成する。
dataset = ImageFolder("train", transform)
#train val test用に分割
#6割　3割　1割
print(len(dataset))
train_len = int(len(dataset)*0.6)
val_len = int(len(dataset)*0.3)
test_len = len(dataset) - train_len - val_len
#分割数の表示
print([train_len,val_len,test_len])
#分割
train_dataset, val_dataset, test_dataset = random_split(dataset=dataset, lengths= [train_len,val_len,test_len], generator=torch.Generator().manual_seed(256))



# DataLoader を作成する。
train_loader = DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True
    )

val_loader = DataLoader(
    val_dataset, 
    batch_size=10,
    shuffle=True
    )








#3.Alexnetを定義
num_classes = 21

'''
バッチサイズ： 10
チャンネル数:3
画像サイズ:128*128
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,512,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(512,256,kernel_size=3)
        self.conv4 = nn.Conv2d(256,128,kernel_size=2)
        self.conv5 = nn.Conv2d(128,64,kernel_size=2)
        self.conv6 = nn.Conv2d(64,32,kernel_size=2)


        self.fc1 = nn.Linear( 3*3*32, 120)
        self.fc2 = nn.Linear(120,60)
        self.fc3 = nn.Linear(60, 21)

    def forward(self, x):
        x = self.conv1(x) #((128 - 3*2)/1)+1 = 123

        x = self.bn1(x)
        x = self.conv2(x) #((123 - 3*2)/1)+1 = 118
        x = self.relu(x)
        x = self.pool(x) #118/2 = 59

        x = self.conv3(x) #((59 - 3)/1)+1 = 57
        x = self.relu(x)
        x = self.pool(x) #57/2 = 28

        x  = self.bn2(x)
        x = self.conv4(x) #((28 - 2)/1)+1 = 27
        x = self.relu(x)
        x = self.pool(x) #27/2 = 13

        x = self.conv5(x) #((13 - 2)/1)+1 = 14
        x = self.relu(x)
        x = self.pool(x) #14/2 = 7

        x = self.conv6(x) #((7 - 2)/1)+1 = 6
        x = self.relu(x)
        x = self.pool(x) #6/2 = 3

        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x




device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net().to(device)


print(net)
print(device)


#loss関数と最適化関数の決定
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.01,momentum=0.9,weight_decay=5e-4)


#学習
#150エポック(学習データ全体を何回繰り返し学習させるか)
num_epochs = 150
best_accuracy = 0
cur_accuracy = 0

#プロット用リスト
train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

for epoch in range(num_epochs):
    #エポックごとに初期化
    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    #train------------------------------------------------
    net.train()
    #ミニバッチで分割して読み込む
    for i,(images,labels) in enumerate(train_loader):
        #view()での変換をしない
        images,labels = images.to(device),labels.to(device)
    
        #勾配をリセット
        optimizer.zero_grad()
        #順伝搬の計算
        outputs = net(images)
        #loss計算
        loss = criterion(outputs,labels)
        #lossのミニバッチ計算
        train_loss += loss.item()
        #accuracyをミニバッチ分溜め込む
        #正解ラベル(labels)と予測値(outputs.max(1))が同じに場合、1
        train_acc += (outputs.max(1)[1] == labels).sum().item()
        #逆伝搬の計算
        loss.backward()
        #重みの更新
        optimizer.step()


    #平均loss,accuracyを計算
    avg_train_loss = train_loss/len(train_loader.dataset)
    avg_train_acc = train_acc/len(train_loader.dataset)

    #val------------------------------------------------
    '''
    valは、trainと何が違うか。
    ・逆伝播の計算と重みの更新を行わない。
    '''



    model = net.eval()
    #評価するときに余計な計算が起こらないようにtorch.no.gradを使用
    with torch.no_grad():
        #ミニバッチで分割して読み込む
        for images,labels in val_loader:
        #view()での変換をしない
            images,labels = images.to(device),labels.to(device)
            #順伝搬の計算
            outputs = net(images)
            #loss計算
            loss = criterion(outputs,labels)
            #lossのミニバッチ計算
            val_loss += loss.item()
            #accuracyをミニバッチ分溜め込む
            #正解ラベル(labels)と予測値(outputs.max(1))が同じに場合、1
            val_acc += (outputs.max(1)[1] == labels).sum().item()
    #平均loss,accuracyを計算
    avg_val_loss = val_loss/len(val_loader.dataset)
    avg_val_acc = val_acc/len(val_loader.dataset)
    
    #検証用データの正答率(val_acc)が過去のものよりも高かったら、ベストなモデルとして保存する。
    cur_accuracy = avg_val_acc
    best_model = copy.deepcopy(model)

    if cur_accuracy > best_accuracy:
        print('current is best_model')
        best_model = copy.deepcopy(model)
        best_accuracy = cur_accuracy
    torch.save(best_model.state_dict(), 'softmax_animal_model.pth')




    #訓練データのlossと検証データのlossとaccuracyをログで出している
    print('Epoch[{}/{}],Loss{loss:.4f},val_loss:{val_loss:.4f},val_acc:{val_acc:.4f}'.format(epoch+1,num_epochs,i+1,loss=avg_train_loss,val_loss = avg_val_loss,val_acc = avg_val_acc))

    #グラフをプロットするようにリストに格納
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)


#結果をプロット
plt.figure()
plt.plot(range(num_epochs),train_loss_list,color = 'blue',label = 'train_loss')
plt.plot(range(num_epochs),val_loss_list,color = 'green',label = 'val_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
plt.savefig('softmax extend Training and validation loss.png')


plt.figure()
plt.plot(range(num_epochs),train_acc_list,color = 'blue',label = 'train_acc')
plt.plot(range(num_epochs),val_acc_list,color = 'green',label = 'val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation acc')
plt.grid()
plt.savefig('softmax extend Training and validation acc.png')


'''
current is best_model
Epoch[116/150],Loss0.1639,val_loss:0.2014,val_acc:0.4196
'''