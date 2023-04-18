from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np




#AlexNet風
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

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

        self.dropout1 = nn.Dropout2d(0.3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #入力20出力10の全結合層
        self.fc1 = nn.Linear(20, 10)
        #dim =1だと行単位でSoftmaxをかけてくれる。
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool1(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.bn4(x)
        x = self.dropout1(x)

        x =self.conv7(x)
        x = self.relu7(x)

        x =self.conv8(x)
        x = self.relu8(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.softmax(x)

        return x




# モデルを読み込む
model = Net()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#モデルの重み読み込む
model.load_state_dict(torch.load("softmax_extend_model.pth"))
#モデルをGPUに送る
model.to(device)
#検証
model.eval()


app = FastAPI()

# 画像を受け取り、予測を返すエンドポイントを定義する
@app.post("/predict")

#並行処理　アップロードされたファイルを受け取る
async def predict(file: UploadFile = File(...)):
    # 画像を読み込み
    image = Image.open(io.BytesIO(await file.read()))
    
    #画像の前処理
    #Composeオブジェクトを呼び出すことで、変換関数が作成され、その後(image)を追加することで、image変数に変換関数を適用する
    image = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])(image)
    
    
    image = image.to(device)
    #画像をミニバッチの形に変換
    image = image.unsqueeze(0)

    # 予測を行う
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)


    '''データの確認'''
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck')  # CIFAR10のクラス



    #torch.max()の範囲は、[-2,1]
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    print('予測：{} 予測スコア：{}'.format(classes[predicted[0]],torch.max(outputs)))

    classfication_result = classes[predicted[0]]
    classfication_value = float(torch.max(outputs))



    # 結果を返す
    return {"predictions":[{"classification_results":[classfication_result],"score":[classfication_value]}]}

