import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
import glob
import torch.utils.data as data
from torchvision import models,transforms


class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                    resize, scale=(0.5, 1.0)),  # データオーギュメンテーション
                transforms.RandomHorizontalFlip(),  # データオーギュメンテーション
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 画像中央をresize×resizeで切り取り
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)
class Animal:
    def __init__(self,animal,image_path,label):
        self.animal=animal
        self.image=image_path
        self.label=label
class AnimalDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list =file_list   # ファイルパスのリスト
        self.transform = transform  # 前処理クラスのインスタンス
        self.phase = phase  # train or valの指定
    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.file_list)
    def __getitem__(self, index):
        '''
        前処理をした画像のTensor形式のデータとラベルを取得
        '''
        # index番目の画像をロード
        img_path = self.file_list[index].image
        # [高さ][幅][色RGB]
        img = Image.open(img_path)
        # 画像の前処理を実施
        img_transformed = self.transform(img, self.phase)
        # torch.Size([3, 224, 224])
        img_resize = img_transformed.numpy().transpose((1, 2, 0))

        img_resize = np.clip(img_resize, 0, 1)
        # 画像のラベルをanimalクラスから抜き出す
        label= self.file_list[index].label
        if label=="Dog":
            label=0
        elif label=="Cat":
            label=1

        #img_resizeをreturnに指定すれば、加工した画像情報も絵で確認可能
        return img_transformed, label
#犬と猫のファイルを訓練用・検証用に8:2の割合で分ける
def Animal_file_list(dog_files,cat_files):
    train_list=list()
    val_list=list()
    for i,dog_file_path in enumerate(dog_files):
        dog_files_len=len(dog_files)
        Dogs=Animal("Dog",dog_file_path,"Dog")
        length=(i+1)/dog_files_len
        if length > 0.8:
            val_list.append(Dogs)
        else:
            train_list.append(Dogs)
    for i,cat_file_path in enumerate(cat_files):
        cat_files_len=len(cat_files)
        Cats=Animal("Cat",cat_file_path,"Cat")
        length=(i+1)/cat_files_len
        if length > 0.8:
            val_list.append(Cats)
        else:
            train_list.append(Cats)
    return train_list,val_list
