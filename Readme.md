# 課題1
ひとまずこれで
C:\Users\rikut\pytorch_learn\Covertype
4/9
これをレポート化


# 課題2
### Computer Vision1
CIFAR-10 datasetを用いて、10クラス(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)分類を行う機械学習モデルを作成し、下記の要件を満たすAPIを作成してください。使用する機械学習モデルやフレームワーク(Flask, FastAPIなど)は自由とします。
request: 画像データまたは、保存先ストレージのURI
response: 与えられた画像の分類結果とその予測スコア
4/9
コアの機械学習の部分
train: C:\Users\rikut\pytorch_learn\CIFAR10_cnn\cifar10_cnn_train_extend.py
predict: C:\Users\rikut\pytorch_learn\CIFAR10_cnn\smart_cifar10_cnn_predict.py
~~fast_apiを使って実装~~
4/10
実装済み
完了

# 課題3
### 問1
cnnを使っても、val_accが0.4196しか出なかったため、ViTでやってみてる

### 問2
yolov5でやってみる。
デフォルト？のcoco128の中のディレクトリ構造に倣ってみる



# 課題4
chABSA-datasetを使用して、ラベルがつけられているものが文中にあれば、そのポジネガラベルに従い、ラベルがつけられてないものしか文中にない場合は、neutralとする
また、chABSA-datasetのみでは不十分であると考えたため、日本語感情表現辞書を使用して、ポジネガ分類を行いました。として、自分が作った既存のポジネガ分類器としたapiの両方を実装する。そこでの違いも伝える。
chabsa_mlpの中にchabsa_datasetをディレクトリごと入れる。
学習用データを作ったので、あとは、学習させる。