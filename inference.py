#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data

from networks import ECO_Lite
from  dataloader import (
    make_datapath_list, get_label_id_dictionary,
    VideoTransform, VideoDataset
)
from prepare import load_pretrained_ECO

def show_eco_inference_result(dir_path, outputs_input, id_label_dict, idx=0):
    """ ミニバッチの各データに対して，推論結果の上位5つを出力する関数 """
    print('ファイル：', dir_path[idx])  # ファイル名

    outputs = outputs_input.clone()     # コピーを作成

    for i in range(5):
        # 1位から5位までを表示
        output = outputs[idx]
        _, pred = torch.max(output, dim=0)  # 確率最大値のラベルを予測
        class_idx = int(pred.numpy())       # クラスIDを出力
        print('予測第{}位：{}'.format(i + 1, id_label_dict[class_idx]))
        outputs[idx][class_idx] = -1000     # 最大値だったものを消す(小さくする)

def main():
    # 画像群のパス作成
    root_path = './data/kinetics_videos/'
    video_list = make_datapath_list(root_path)

    # 動画クラスラベルとIDの辞書作成
    label_dicitionary_path = './video_download/kinetics_400_label_dicitionary.csv'
    label_id_dict, id_label_dict = get_label_id_dictionary(label_dicitionary_path)

    # 前処理の設定
    resize, crop_size = 224, 224
    mean, std = [104, 117, 123], [1, 1, 1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    # Datasetの作成
    val_dataset = VideoDataset(video_list, label_id_dict, num_segments=16,
                               phase='val', transform=video_transform,
                               img_template='image_{:05d}.jpg')

    # DataLoaderを作成して，データをロード
    batch_size = 8
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
    batch_iterator = iter(val_dataloader)  # イテレータに変換
    imgs_transformeds, labels, label_ids, dir_path = next(batch_iterator)

    # モデルをインスタンス化し，推論モードに変更後，学習済みモデルをロード
    net = ECO_Lite()
    net.eval()
    net.load_state_dict(torch.load('./models/pretrained.pth'))

    # ECOで推論
    with torch.set_grad_enabled(False):
        outputs = net(imgs_transformeds)

    # 予測を実施
    idx = 0
    show_eco_inference_result(dir_path, outputs, id_label_dict, idx)

if __name__ == '__main__':
    main()
