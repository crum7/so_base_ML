from transformers import BertJapaneseTokenizer,BertModel
import torch
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


json_file = open('train.json', 'r')

data = pd.read_json(json_file, orient='records', lines=True,encoding='utf-8_sig')

sentence_1 = data["sentence1"]
sentence_2 = data["sentence2"]

for i in range(0,len(sentence_1)):
    text1 = sentence_1[i]
    text2 = sentence_2[i]

    encoded_dict1 = tokenizer.encode_plus(
                            text1,                      # 文章をエンコード
                            add_special_tokens = True,  # 特殊トークンを追加
                            max_length = 64,           # 最大のトークン数
                            pad_to_max_length = True,   # 不足分をパディング
                            return_attention_mask = True, # Attention maskを作成
                            return_tensors = 'pt',      # PyTorch Tensorを返す
                    )
    input_ids1 = encoded_dict1['input_ids']
    attention_masks1 = encoded_dict1['attention_mask']

    encoded_dict2 = tokenizer.encode_plus(
                            text2,                      # 文章をエンコード
                            add_special_tokens = True,  # 特殊トークンを追加
                            max_length = 64,           # 最大のトークン数
                            pad_to_max_length = True,   # 不足分をパディング
                            return_attention_mask = True, # Attention maskを作成
                            return_tensors = 'pt',      # PyTorch Tensorを返す
                    )
    input_ids2 = encoded_dict2['input_ids']
    attention_masks2 = encoded_dict2['attention_mask']

    with torch.no_grad():
        last_hidden_states1 = model(input_ids1, attention_mask=attention_masks1)
        last_hidden_states2 = model(input_ids2, attention_mask=attention_masks2)

    features1 = last_hidden_states1[0][:,0,:].numpy() # [CLS]トークンの特徴量を取得
    features2 = last_hidden_states2[0][:,0,:].numpy() # [CLS]トークンの特徴量を取得


    #各相関係数
    pearson_value, _ = pearsonr(features1[0], features2[0])
    spearman_value, _ = spearmanr(features1[0], features2[0])

    f = open('result.json', 'a')

    f.write('{“metrics”: pearson, “score”: '+str(pearson_value*5)+'}\n')
    f.write('{“metrics”: spearman, “score”: '+str(spearman_value*5)+'}\n')


    f.close()
