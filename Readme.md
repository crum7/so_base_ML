# covertypeデータセットを使って、気候や標高などの環境条件から、森林を占める木の種類を予測する多クラス分類問題
### データ整理
covtype.dataをcsvの形にし、左からそれぞれに
Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40,Cover_Type  
の名前を割り当てた。そして、目的変数であるCover_Typeとそれ以外をそれぞれtrain_yとtrain_xに分けた。またその中で検証用に分けた。  

使用したモデル：決定木  
結果：平均絶対残差：0.3
考察：決定木モデルの精度が低かった原因は、データ品質の悪さ、データ量の不足、適切なハイパーパラメータの設定不足などが考えられる。これらの問題に対処するためには、データの前処理、データ拡張、ハイパーパラメータのチューニングなどを行うことが必要である。また、ランダムフォレストなどを利用し、比較的精度の低い決定木をバギングさせることで高い精度を出すことも解決策なのではないか。  

使用したモデル：ランダムフォレスト  
結果：平均絶対残差： 0.4
考察：決定木よりも精度が高くなった。決定木よりランダムフォレストの方が精度が高くなった理由としては、ランダムフォレストは、決定木を使用しバギングを行っているためである。また、気をつけたポイントとしてはランダムフォレストは過学習が起きやすいため、コード内のget_mae関数を使用し、一番正解率が高い葉の数を調査した。その結果、葉の枚数が50の時に一番正解率が高かった。  


# CIFAR-10 datasetを用いて、10クラス分類とAPIの作成
### データの加工
torchvision.datasets.CIFAR10を使用して、データの収集を行った。  
そして、torchvision.transforms.RandomAffine・RandomHorizontalFlip・Normalizeを利用して、  
データの拡張と正規化をtrain用・val用のデータセットそれぞれに行った。  

### 学習
CIFAR-10を用いて、8層の畳み込み層を利用した学習・推定を行った。  
当初は、以下のMLP(3層の全結合層)を用いて行っていたのだが、50%程度の正解率に収まってしまったこと且つエポック数を増やすとすぐに過学習が起きてしまっていた。  

MLP model
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet,self).__init__()
        #画像を1次元に変換
        self.fc1 = nn.Linear(32*32*3,600)
        self.fc2 = nn.Linear(600,600)
        self.fc3 = nn.Linear(600,num_classes)
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout2d(0.2)
    
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout2(x)
        return F.relu(self.fc3(x))


そのため、CNNで行うことにした。  
全体のコードは添付してある通りである。cnnはALexnetの形を意識し、その後正解率に応じて、自分で各種層を付け足した。  
そして最終的には、以下のようなモデルを構築し、学習させた。  
バッチサイズ：64  
チャンネル数：3  

入力: 3チャンネルの2次元画像  
出力: 10クラスの分類  
特徴抽出器層:  
1つ目の畳み込み層: 64フィルター, カーネルサイズ11, ストライド4, パディング5  
ReLU活性化関数  
バッチ正規化  
2つ目の畳み込み層: 600フィルター, カーネルサイズ5, パディング2  
ReLU活性化関数  
バッチ正規化  
3つ目の畳み込み層: 400フィルター, カーネルサイズ3, パディング1  
ReLU活性化関数  
バッチ正規化  
4つ目の畳み込み層: 200フィルター, カーネルサイズ3, パディング1  
ReLU活性化関数  
5つ目の畳み込み層: 100フィルター, カーネルサイズ3, パディング1  
ReLU活性化関数  
6つ目の畳み込み層: 80フィルター, カーネルサイズ3, パディング1  
ReLU活性化関数  
バッチ正規化  
7つ目の畳み込み層: 40フィルター, カーネルサイズ3, パディング1  
ReLU活性化関数  
8つ目の畳み込み層: 20フィルター, カーネルサイズ3, パディング1  
ReLU活性化関数  
2Dドロップアウト (割合0.25)  
最大プーリング層: カーネルサイズ2, ストライド2  
全結合層:  
入力: 20  
出力: 10  
ソフトマックス活性化関数  





# Animals Detection Images Datasetを使用した21クラスの分類問題
### データの加工
kaggleの「Animals Detection Images Dataset」を使用して、データの収集を行った。  
また、開発していた時は、21クラスの分類であったため、Bear・Brown・Bull・Butterfly・Camel・Canary・Caterpillar  
Cattle・Centipede・Cheetah・Chicken・Crab・Crocodile・Deer・Duck・Eagle・Elephant・Fish・Fox・Frog・Giraffeの21クラスの分類を行った。  
また、いくつかのデータを見たところ写真のサイズがバラバラであったため、画像のサイズを揃える必要があった。  
そのため、transforms.Resize()を使用して、3*128*128に加工した。  
そして、torchvision.transforms.RandomAffine・RandomHorizontalFlip・Normalizeを利用して、  
データの拡張と正規化をtrain用・val用のデータセットそれぞれに行った。  
また、動物の写真がディレクトリごとに分かれて入っていたため、torchvision.datasets.ImageFolderを使用し、datasetを作成した。  
それをtorch.utils.data.random_splitを使用し、train/val/testに分けた。  

### 学習
当初、6層の畳み込み層・3層の全結合層を使用した。以下のモデルを作成した。  
バッチサイズ：10  
チャンネル数：3  
画像サイズ：128*128  

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

しかし、このモデルでは21種類の分類に耐えることが出来ず、正解率は41.96%に留まった。  
そのためViT(Vision Transfomer)を使用し、学習・推論させることにした。  
vision transfomerを使用するために、vit_pytorchを使用した。本来ならば、モデルを自分で構築したかったのだが、自分の力不足故にまだtransfomerのモデルを期間内に自作出来なかったため、今回は既存のシンプルなライブラリを使用して行った。  

### 検証
ViTモデルでの正解率は44.23%とcnnのモデルから約3%上昇した。  



# chABSA-datasetを用いて、各業績概要のテキストをpositive, negative, neutralのいずれかに分類する機械学習モデルを作成、apiの作成
### データの加工
chABSA-datasetの特徴として、文中のどの単語がポジティブなのかネガティブなのかは示されているが、それらは直接文のポジティブ・ネガティブ度合いには関係ないものになってしまっている。そのため、まず文にラベルをつける作業から行った。  
ラベルがつけられているものがopinions中にあれば、そのポジネガのラベルに従い文のポジネガを決定し、文中にない場合は、その文をneutralとした。  
具体的には、opinionsの中にpositive、negativeが含まれている場合、  
positiveだったら、+1点  
negativeだったら、-1点  
neutralだったら、0点と点数をつけ、その合計点数に応じて文のポジティブ・ネガティブ・ニュートラルを決定する。  
合計点数が正だった場合はポジティブとし、0だった場合はニュートラル。負だった場合は、ネガティブとした。  
また、opinonsの中に何も含まれていない場合は、文をneutralとした。  
また、csvの形に変更し、学習時に取り扱いやすい形とした。  
上で文をネガティブ・ニュートラル・ポジティブと判別したのちに、それぞれの文に0,1,2とラベル付けを行った。  
次に、データをtrain用とval用に分割した。この時、文をデフォルトの会社順ではなくバラバラに並べ、7:3の割合でtrain用csvの末尾からval用のcsvに移動した。  

### 学習
ローカルでRTX 3070を使用し学習させた。  
bertのhuggingfaceで公開されているcl-tohoku/bert-base-japanese-whole-word-maskingを使用し、ファインチューニングを行った。  
batch数：3  
エポック数：7  
weight_decay = 0.01  
batch_sizeが3と比較的小さいのは、これ以上batch_sizeを大きくすると、RTX 3070のRAM容量(8GB)を超えてしまい、学習ができなくなってしまうためである。  
また、エポック数はいくつか試した上で最終的に7という数字に落ち着いている。  
最終的に、検証用のf1スコアは、88.599%という比較的高いF値を出せた。

# pearson,spearman相関係数
## データの加工
JGLUE-STS内のtrain-v1.1.jsonから、pandasを利用して、sentence1,sentence2を取得した。それらをpandas.seriesの形にまとめ、まとめたものをfor文で回した。

## 相関係数の演算
//bertを使用してtensorに直したという説明をする
その後、トークン化した2文を引数にとり、scipyのperasonrでPearson相関係数を計算し、scipyのspearmanrでSpearman相関係数を演算した。また、0-5の範囲内で示すため各結果に5をかけた。
## 出力
それぞれの相関係数の種類と結果を提示されているフォーマットに従い、jsonlの形で書き出した。








