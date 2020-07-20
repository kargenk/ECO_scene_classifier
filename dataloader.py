#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torchvision

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

if __name__ == '__main__':
    root_path = './data/kinetics_videos/'
    video_list = make_datapath_list(root_path)
    
    print(video_list[0])
    print(video_list[1])