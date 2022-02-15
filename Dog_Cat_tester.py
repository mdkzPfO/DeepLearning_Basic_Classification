import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
import glob
import torch.utils.data as data
from torchvision import models,transforms
import torch.nn as nn

class Animal:
    def __init__(self,animal,image_path,label):
        self.animal=animal
        self.image=image_path
        self.label=label

class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform =transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
    def __call__(self, img):
        return self.data_transform(img)

class AnimalDataset(data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list =file_list   # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)
    def __getitem__(self):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''
        # index番目の画像をロード
        img_path = self.file_list.image
        # [高さ][幅][色RGB]
        img = Image.open(img_path)
        # 画像の前処理を実施
        img_transformed = self.transform(img)
        # torch.Size([3, 224, 224])
        img_resize = img_transformed.numpy().transpose((1, 2, 0))

        img_resize = np.clip(img_resize, 0, 1)
        # 画像のラベルをanimalクラスから抜き出す
        label= self.file_list.label
        if label=="Dog":
            label=0
        elif label=="Cat":
            label=1

        #img_resizeをreturnに指定すれば、加工した画像情報も絵で確認可能
        return img_transformed, label
#画像の前処理に必要なスコアを記載
size = 224
#why????
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

Animal_files = "./data/test_dog/puppy.jpg"
Animal_files=Animal("Dog",Animal_files,"Dog")
Animal_transformed=AnimalDataset(file_list=Animal_files, transform=ImageTransform(size, mean, std))

#ラベルが犬になっていることを確認
test_pets=Animal_transformed.__getitem__()[0]
test_label=Animal_transformed.__getitem__()[1]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

use_pretrained=True
net=models.vgg16(pretrained=use_pretrained)
net.classifier[6]=nn.Linear(in_features=4096,out_features=2)
model2 = net.to(device)
PATH="weights_fine_tuning.pth"

model2.load_state_dict(torch.load(PATH))
model2.eval()
inputs = test_pets.to(device)
inputs=inputs.unsqueeze(0)
labels=np.array(test_label)
labels = torch.from_numpy(labels)
labels = labels.to(device)
outputs = model2(inputs)
print("結果")
_, preds = torch.max(outputs, 1)
if preds==0:
    print("Dog")
else:
    print("Cat")
