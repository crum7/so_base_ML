import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

#データの概要
covertype_file_path = './covtype.csv'
covertype_data = pd.read_csv(covertype_file_path)
print(covertype_data.head())

#データ分割
base_colums = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']
x = covertype_data[base_colums]
y = covertype_data.Cover_Type

train_x,val_x,train_y,val_y = train_test_split(x,y,random_state=0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
#ベストな数を探した

for i in candidate_max_leaf_nodes:
    best_mae = get_mae(i,train_x,val_x,train_y,val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(i, best_mae))


best_decision_tree_size = 50
#条件木モデル
final_model = DecisionTreeRegressor(max_leaf_nodes=best_decision_tree_size, random_state=1)


#ランダムフォレストモデル
#final_model = RandomForestRegressor(max_leaf_nodes =3,random_state=1)
print(final_model)
print(final_model.fit(train_x, train_y))
val_prediction = final_model.predict(val_x)

print("正解値：{}  予測値：{}".format(val_y[:5],val_prediction[:5]))
print(mean_absolute_error(val_y, val_prediction))

