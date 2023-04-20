import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import copy

'''
CIFAR-10を畳み込みニューラルネットワーク(CNN)で画像分類
*AlexNetを模倣
Loss関数にCrossEntropyを使用し、最適化関数にSGDを使用
Conv2dの入力は(バッチサイズ、チャンネル数、縦、横)のTensor

*AlnexNetとは、2012年のILSVRC優勝したモデル
5層の畳み込みと3層の全結合層で構成されたもの

'''

#1.データの収集

#transformする項目をここで決める。
#transforms.ToTensor()を忘れずに
transform = transforms.Compose([
    transforms.RandomAffine([0,30], scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


#train_datasetを設定し、train用のデータを読み込む
train_dataset = torchvision.datasets.CIFAR10(
    root = './data',
    train=True,
    transform=transform
    )

#test_datasetを設定し、test用のデータを読み込む（train = falseになっているところポイント）
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    transform=transform
    )






#2.データの読み込み
#DataLoaderは、バッチサイズの画像と正解ラベルをdatasetから取り出す
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = 64,
    shuffle = True,
    num_workers=2
)

#test_loaderは、train_loaderとは異なり、shuffleはしない。
test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = 64,
    shuffle = False,
    num_workers=2
)
#DataLoaderの定義より、imagesは、(バッチサイズ：64,チャンネル数：3,縦：32,横：32)のTensor形状になる。
#images[0]には、（チャンネル数：3,縦：32,横：32）のTensor
#labels[0]には、正解ラベルが入る
#Dataloaderでバッチサイズごとにまとまったデータを返しながら、学習や検証を進めていく



#3.Alexnetを定義
num_classes =10

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # feature extractor layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 600, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(600)

        self.conv3 = nn.Conv2d(600, 400, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(400)


        self.conv4 = nn.Conv2d(400, 200, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(200, 100, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(100, 80, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(80)

        self.conv7 = nn.Conv2d(80, 40, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(40, 20, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout2d(0.25)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #入力30出力256の全結合層
        self.fc1 = nn.Linear(20, 10)
        #dim =1だと行単位でSoftmaxをかけてくれる。
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
       # x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
       # x = self.dropout1(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool1(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn4(x)
       # x = self.dropout1(x)

        x =self.conv7(x)
        x = self.relu7(x)

        x =self.conv8(x)
        x = self.relu8(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)

        return x
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = AlexNet().to(device)

print(net)


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
        for images,labels in test_loader:
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
    avg_val_loss = val_loss/len(test_loader.dataset)
    avg_val_acc = val_acc/len(test_loader.dataset)
    
    #検証用データの正答率(val_acc)が過去のものよりも高かったら、ベストなモデルとして保存する。
    cur_accuracy = avg_val_acc
    best_model = copy.deepcopy(model)

    if cur_accuracy > best_accuracy:
        print('current is best_model')
        best_model = copy.deepcopy(model)
        best_accuracy = cur_accuracy
    torch.save(best_model.state_dict(), 'softmax_extend_model.pth')




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


