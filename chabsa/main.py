from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer,BertTokenizer, BertForSequenceClassification
import csv
import re
from fastapi import FastAPI, File, UploadFile


# パイプラインの準備
model = AutoModelForSequenceClassification.from_pretrained('posizga_chabsa_classification') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)



app = FastAPI()


# 画像を受け取り、予測を返すエンドポイントを定義する
@app.post("/predict")

#並行処理　アップロードされたファイルを受け取る
async def predict(text: str):

    #推論
    print(text)
    result = classifier(text)
    print(result[0]['label'])

    if result[0]['label'] == 'LABEL_2':
        label = 'positive'
    elif result[0]['label'] == 'LABEL_1':
        label = 'neutral'
    elif result[0]['label'] == 'LABEL_0':
        label = 'negative'


    '''
    LABEL_2がポジティブ
    LABEL_1がニュートラル
    LABEL_0がネガティブ
    '''

    return {"predictions":[{"classification_results":[label],"score":[result[0]['score']]}]}

