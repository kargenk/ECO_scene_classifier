#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

def make_datapath_list(root_path):
    """
    動画を画像データにしたディレクトリへのパスのリストを作成する．
    
    Inputs
    ----------
    root_path : str
        データディレクトリへのrootパス
    
    Returns
    ----------
    video_list : list(str)
        動画を画像データにしたディレクトリへのパスのリスト
    """

    video_list = list()
    class_list = os.listdir(path=root_path)

    # 各クラスの動画ファイルを画像化したディレクトリへのパスを取得
    for class_list_i in class_list:
        class_path = os.path.join(root_path, class_list_i)  # クラスのディレクトリへのパス

        # 各クラスのディレクトリ内の画像を取得
        for file_name in os.listdir(class_path):
            name, ext = os.path.splitext(file_name)  # ファイル名と拡張子

            # 元の動画(.mp4)は無視
            if ext == '.mp4':
                continue
            
            # 動画を画像に分割して保存したディレクトリのパスを追加
            video_img_directory_path = os.path.join(class_path, name)
            video_list.append(video_img_directory_path)
    
    return video_list

def get_label_id_dictionary(label_dicitionary_path='./video_download/kinetics_400_label_dicitionary.csv'):
    """
    Kinetics-400のラベル名をIDに変換する辞書と，逆にIDをラベル名に変換する辞書を返す関数．
    
    Inputs
    ----------
    label_dictionary_path : str
        Kinetics-400のクラスラベル情報のcsvファイルへのパス
    
    Returns
    ----------
    label_id_dict : dict()
        ラベル名をIDに変換する辞書
    id_label_dict : dict()
        IDをラベル名に変換する辞書
    """

    label_id_dict = dict()
    id_label_dict = dict()

    with open(label_dicitionary_path, encoding='utf-8_sig') as f:

        # 読み込む
        reader = csv.DictReader(f, delimiter=',', quotechar='"')

        # 1行ずつ読み込み，辞書型変数に追加
        for row in reader:
            label_id_dict.setdefault(
                row['class_label'], int(row['label_id']) - 1)
            id_label_dict.setdefault(
                int(row['label_id']) - 1, row['class_label'])

    return label_id_dict, id_label_dict

class GroupResize():
    """
    画像群をまとめてリサイズするクラス．
    画像の短い方の辺がresizeに変換される(アスペクト比はそのまま)．
    """

    def __init__(self, resize, interpolation=Image.BILINEAR):
        self.rescaler = transforms.Resize(resize, interpolation)
    
    def __call__(self, img_group):
        return [self.rescaler(img) for img in img_group]

class GroupCenterCrop():
    """
    画像群をまとめてクリッピングするクラス．
    (crop_size, crop_size)の画像を切り出す．
    """

    def __init__(self, crop_size):
        self.center_crop = transforms.CenterCrop(crop_size)

    def __call__(self, img_group):
        return [self.center_crop(img) for img in img_group]

class GroupToTensor():
    """ 画像群をまとめてtorch.tensorに変換するクラス． """

    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, img_group):
        # 学習済みデータの形式に合わせるため，[0, 255]で扱う
        return [self.to_tensor(img) * 255 for img in img_group]

class GroupImgNormalize():
    """ 画像群をまとめて標準化するクラス． """

    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img_group):
        return [self.normalize(img) for img in img_group]

class Stack():
    """
    画像群を1つのtensorにまとめるクラス．
    
    Inputs
    ----------
    img_group : list(torch.tensor)
        torch.Size([3, 224, 224])を要素とするリスト
    """

    def __call__(self, img_group):
        # 元の学習データがBGRのため，x.flip(dims=[0])で色チャネルをRGB -> BGRに変換
        # unsqueeze(dim=0)でframes用の次元を追加して，frames次元で結合
        ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0)
                         for x in img_group], dim=0)

class VideoTransform():
    """
    動画を画像にしたファイルの前処理クラス．学習と推論時で異なる動作をする．
    動画を画像に分割しているため，分割された画像群をまとめて前処理する．
    """

    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                # DataAugumentation()  # 今回は無し
                GroupResize(int(resize)),          # 画像群をまとめてリサイズ
                GroupCenterCrop(crop_size),        # 画像群をまとめて切り抜き
                GroupToTensor(),                   # torch.tensorに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()                            # frames次元で結合
            ]),
            'val' : transforms.Compose([
                GroupResize(int(resize)),          # 画像群をまとめてリサイズ
                GroupCenterCrop(crop_size),        # 画像群をまとめて切り抜き
                GroupToTensor(),                   # torch.tensorに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()                            # frames次元で結合
            ])
        }
    
    def __call__(self, img_group, phase):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモード指定フラグ
        """
        return self.data_transform[phase](img_group)

if __name__ == '__main__':
    
    # データパス作成のテスト
    root_path = './data/kinetics_videos/'
    video_list = make_datapath_list(root_path)
    print(video_list[0])
    print(video_list[1])

    # 動画クラスラベルの確認
    label_dicitionary_path = './video_download/kinetics_400_label_dicitionary.csv'
    label_id_dict, id_label_dict = get_label_id_dictionary(label_dicitionary_path)
    print(label_id_dict)