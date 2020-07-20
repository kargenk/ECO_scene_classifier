#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
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
        return ret

class VideoTransform():
    """
    動画を画像にしたファイルの前処理クラス．学習と推論時で異なる動作をする．
    動画を画像に分割しているため，分割された画像群をまとめて前処理する．
    """

    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                # DataAugumentation()  # 今回は無し
                GroupResize(int(resize)),      # 画像群をまとめてリサイズ
                GroupCenterCrop(crop_size),    # 画像群をまとめて切り抜き
                GroupToTensor(),               # torch.tensorに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()                        # frames次元で結合
            ]),
            'val' : transforms.Compose([
                GroupResize(int(resize)),      # 画像群をまとめてリサイズ
                GroupCenterCrop(crop_size),    # 画像群をまとめて切り抜き
                GroupToTensor(),               # torch.tensorに
                GroupImgNormalize(mean, std),  # データを標準化
                Stack()                        # frames次元で結合
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

class VideoDataset(data.Dataset):
    """
    動画のDataset．

    Attributes
    ----------
    video_list : list(str)
        動画を画像群に変換したディレクトリパスのリスト
    label_id_dict : dict()
        クラスラベル名をIDに変換する辞書
    num_segments : int
        動画を何分割して使用するか
    phase : 'train' or 'val'
        (訓練 or 推論) のモードを管理するフラグ
    transform : object
        前処理クラス
    img_template : str
        読み込みたい画像群のファイル名テンプレート
    """

    def __init__(self, video_list, label_id_dict, num_segments,
                 phase, transform, img_template='image_{:05d}.jpg'):
        self.video_list = video_list
        self.label_id_dict = label_id_dict
        self.num_segments = num_segments
        self.phase = phase
        self.transform = transform
        self.img_template = img_template

    def __len__(self):
        ''' 動画の数を返す '''
        return len(self.video_list)

    def __getitem__(self, index):
        ''' 前処理した画像群のデータとラベル，ラベルID，パスを返す '''
        imgs_transformed, label, label_id, dir_path = self.pull_item(index)
        return imgs_transformed, label, label_id, dir_path

    def pull_item(self, index):
        ''' 前処理した画像群のデータとラベル，ラベルID，パスを返す '''

        # 画像群を読み込む
        dir_path = self.video_list[index]
        indices = self._get_indices(dir_path)
        img_group = self._load_imgs(dir_path, self.img_template, indices)

        # ラベルをIDに変換
        # label = (dir_path.split('/')[3].split('/')[0])  # for Ubuntu
        label = dir_path.split('/')[3].split('\\')[0]  # for Windows
        label_id = self.label_id_dict[label]

        # 前処理を実行
        imgs_transformed = self.transform(img_group, phase=self.phase)

        return imgs_transformed, label, label_id, dir_path

    def _load_imgs(self, dir_path, img_template, indices):
        ''' 画像群をまとめて読み込み，リスト化する関数． '''

        img_group = list()
        for idx in indices:
            # 画像を読み込んでリストに追加
            file_path = os.path.join(dir_path, img_template.format(idx))
            img = Image.open(file_path).convert('RGB')
            img_group.append(img)
        
        return img_group

    def _get_indices(self, dir_path):
        ''' 動画全体をself.num_segmentsに分割した際に取得する動画のidxのリストを返す '''

        # 動画のフレーム数
        file_list = os.listdir(path=dir_path)
        num_frames = len(file_list)

        # 動画を取得する間隔幅
        tick = (num_frames) / float(self.num_segments)

        # 動画を間隔幅:tickで取り出す際のidxのリスト
        indices = np.array([int(tick / 2.0 + tick * x)
                            for x in range(self.num_segments)]) + 1
        
        # 例: 250 frame で 16 frame 抽出の場合，
        # tick = 250 / 16 = 15.625
        # indices = [8 24 40 55 71 86 102 118 133 149 165 180 196 211 227 243]

        return indices

if __name__ == '__main__':
    
    # データパス作成
    root_path = './data/kinetics_videos/'
    video_list = make_datapath_list(root_path)
    # print(video_list[0])
    # print(video_list[1])

    # 動画クラスラベルとIDの取得
    label_dicitionary_path = './video_download/kinetics_400_label_dicitionary.csv'
    label_id_dict, id_label_dict = get_label_id_dictionary(label_dicitionary_path)
    # print(label_id_dict)

    # 前処理の設定
    resize, crop_size = 224, 224
    mean, std = [104, 117, 123], [1, 1, 1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    # Datasetの作成
    val_dataset = VideoDataset(video_list, label_id_dict, num_segments=16,
                               phase='val', transform=video_transform,
                               img_template='image_{:05d}.jpg')

    # データの取り出しテスト
    index = 0
    print(val_dataset.__getitem__(index)[0].shape)  # 画像群のtensor
    print(val_dataset.__getitem__(index)[1])        # ラベル名
    print(val_dataset.__getitem__(index)[2])        # ラベルID
    print(val_dataset.__getitem__(index)[3])        # 動画へのパス
    print()

    # DataLoaderにしてテスト
    batch_size = 8
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
    
    batch_iterator = iter(val_dataloader)  # イテレータに変換
    imgs_transformeds, labels, label_ids, dir_path = next(batch_iterator)
    print(imgs_transformeds.shape)
