#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from networks import ECO_Lite

def load_pretrained_ECO(model_dict, pretrained_model_dict):
    """
    ECOの学習済みモデルをロードする関数．
    今回構築したECOは学習済みモデルとレイヤーの順番は同じだが名前が異なる．
    """

    # 現在のネットワークモデルのパラメータ名
    param_names = []
    for name, param in model_dict.items():
        param_names.append(name)

    # 現在のネットワークの情報をコピーして新たなstate_dictを作成
    new_state_dict = model_dict.copy()

    # 新たなstate_dictに学習済みの値を代入
    print('学習済みのパラメータをロードします')
    for index, (key_name, value) in enumerate(pretrained_model_dict.items()):
        name = param_names[index]     # 現在のネットワークでのパラメータ名を取得
        new_state_dict[name] = value  # 値を入れる

        # 何から何にロードされたのかを表示
        print(str(key_name) + ' → ' + str(name))

    return new_state_dict

if __name__ == '__main__':
    # モデルのインスタンス化
    net = ECO_Lite()
    net.eval()

    # 学習済みモデルをロード
    net_model_ECO = './models/ECO_Lite_rgb_model_Kinetics.pth.tar'
    pretrained_model = torch.load(net_model_ECO, map_location='cpu')
    pretrained_model_dict = pretrained_model['state_dict']
    # （注釈）
    # pthがtarで圧縮されているのは，state_dict以外の情報も一緒に保存されているため．
    # そのため読み込むときは辞書型変数になっているので['state_dict']で指定する．

    # 現在のモデルの変数名などを取得
    model_dict = net.state_dict()

    # 学習済みモデルのstate_dictを取得
    new_state_dict = load_pretrained_ECO(model_dict, pretrained_model_dict)

    # 学習済みモデルのパラメータを代入
    net.eval()  # ECOネットワークを推論モードに
    net.load_state_dict(new_state_dict)

    # ロードした重みを保存
    torch.save(net.state_dict(), './models/pretrained.pth')
