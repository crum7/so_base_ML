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
from vit_pytorch import ViT


'''
画像に映る動物の種類をvit(Vision Transformer)で21クラスに分類する
'''

#vision Trabsformerは、vit-pytorchを使用する

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


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Vision Transformerモデル
net = ViT(
    image_size=128,
    patch_size=4,
    num_classes=21,
    dim=256,
    depth=3,
    heads=4,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1
).to(device)


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
plt.savefig('vit Training and validation loss.png')


plt.figure()
plt.plot(range(num_epochs),train_acc_list,color = 'blue',label = 'train_acc')
plt.plot(range(num_epochs),val_acc_list,color = 'green',label = 'val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Training and validation acc')
plt.grid()
plt.savefig('vit Training and validation acc.png')


'''
current is best_model
Epoch[126/150],Loss0.1659,val_loss:0.1889,val_acc:0.4423
'''