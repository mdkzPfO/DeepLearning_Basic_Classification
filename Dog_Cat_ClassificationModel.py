import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
import glob
import torch.utils.data as data
from torchvision import models,transforms
import dataset
from dataset import Animal_file_list,AnimalDataset,Animal,ImageTransform
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
print(torch.cuda.is_available())

###データセットを作成する
##画像データのファイルパスを格納する
##対象フォルダのファイルパスを上から順番に取得する
dog_files = glob.glob("data/dog/*")
cat_files = glob.glob("data/cat/*")
#画像の前処理に必要なスコアを記載
size = 224
#why????
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
#犬及び猫の画像データ、ラベルをクラスをAnima_file_listクラスに格納する
#さらに格納を行う際に訓練リストと検証リストで分ける
train_list,val_list=Animal_file_list(dog_files,cat_files)
#データの前処理を実施する
train_dataset =  AnimalDataset(
    file_list=train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset =  AnimalDataset(
    file_list=val_list, transform=ImageTransform(size, mean, std), phase='val')
#動作確認
index= 10
test=train_dataset.__getitem__(index)[0].size()
print(test)
test=train_dataset.__getitem__(index)[1]
print(test)
test=val_dataset.__getitem__(index)[0].size()
print(test)
test=val_dataset.__getitem__(index)[1]
print(test)
##DataLoaderを作成する
##ミニバッチのサイズを指定
batch_size = 32

#DataLoaderを作成
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False)

print(train_dataloader)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作確認
batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
inputs, labels = next(
    batch_iterator)  # 1番目の要素を取り出す
print("重要")
print(inputs.size())
print(labels)
print("重要終わり")

###VGGモデルを定義する
use_pretrained=True
#ImageNetで学習済みのvggモデルによる転移学習を行う
net=models.vgg16(pretrained=use_pretrained)

#分類を行う為のclassifierのみ変更する
net.classifier[6]=nn.Linear(in_features=4096,out_features=2)
##vggモデルを訓練モードに切り替える
net.train()
#損失関数を定義する
criterion=nn.CrossEntropyLoss()

#最適化手法を定義する
# ファインチューニングで学習させるパラメータを、変数params_to_updateの1～3に格納する
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# 学習させる層のパラメータ名を指定
#特徴量の抽出に関するパラメーターは全てパラメーターをアップデートする
update_param_names_1 = ["features"]
#分類に関するパラメーターのうちLinearレイヤをアップデートする
update_param_names_2 = ["classifier.0.weight",
                        "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

# パラメータごとに各params_to_updateリストに格納する
#ここではupdate_param_names_1で記載したレイヤと合致するレイヤがVGG16にあれば、格納を行っている
for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1に格納：", name)

    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2に格納：", name)

    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3に格納：", name)

    else:
        param.requires_grad = False
        print("勾配計算なし。学習しない：", name)

#各パラメーターごとに学習率を決め、SGDによる勾配降下インスタンスoptimizerを作成する
#momentumを設定することでなめらかな最適化を実現する。ここで0.9が指定されているのはmomentumにおける最適値とされているため
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)

###学習と検証用の関数を作成する
##引数として、vggモデル、訓練・検証データがまとまった辞書、損失関数、パラメーターの初期値を食わせた最適化手法、処理回数(エポック)
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    epoch_acc_list_train=list()
    epoch_acc_list_val=list()
    epoch_list=list()

    # 初期設定
    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # epoch数分のループを開始する
    for epoch in range(num_epochs):
        #epoch数が1からはじまるリストを作る。処理状況把握の際に表示されるepoch数が分かりやすくなるので。
        epoch_list.append(epoch+1)
        #処理回数を各エポックループごとに表示する
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        #訓練データからループを開始する
        for phase in ['train', 'val']:
            #初回は訓練モードで学習が開始される
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに
            #????
            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略

            if (epoch == 0) and (phase == 'train'):
                continue
            if (epoch == 1) and (phase == 'train'):
                epoch_acc_list_train.append(epoch_loss)

            # データローダーからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(device)
                labels = labels.to(device)

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)  # 損失を計算
                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # 結果の計算
                    epoch_loss += loss.item() * inputs.size(0)  # lossの合計を更新
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)
            if phase=="train":
                epoch_acc_list_train.append(epoch_loss)
            elif phase=="val":
                epoch_acc_list_val.append(epoch_loss)
    return epoch_list,epoch_acc_list_train,epoch_acc_list_val
###学習精度をmatplotlibでグラフ化する
num_epochs=5
epoch_list,epoch_acc_list_train,epoch_acc_list_val=train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
fig, ax = plt.subplots()
c1,c2= "blue","green" # 各プロットの色
l1,l2= "Train","Val" # 各ラベル
print(epoch_list)
print(epoch_acc_list_train)
print(epoch_acc_list_val)
y1=epoch_acc_list_train
y2=epoch_acc_list_val
#y1=list()
#for i in epoch_acc_list_train:
#    num=i.numpy()
#    y1.append(num)
#y2=list()
#for i in epoch_acc_list_val:
#    num=i.numpy()
#    y2.append(num)
print(y1)
print(y2)
ax.set_xlabel('x')  # x軸ラベル
ax.set_ylabel('y')  # y軸ラベル
ax.grid()            # 罫線
#ax.set_xlim([-10, 10]) # x方向の描画範囲を指定
#ax.set_ylim([0, 1])    # y方向の描画範囲を指定
ax.plot(epoch_list,y1, color=c1, label=l1)
ax.plot(epoch_list,y2, color=c2, label=l2)
ax.legend(loc=0)    # 凡例
plt.show()
save_path='./weights_fine_tuning.pth'
torch.save(net.state_dict(),save_path)
