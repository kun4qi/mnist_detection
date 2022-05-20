#mnistの1を正常画像、0を異常画像
train.pyで学習させてresultフォルダにモデルを保存、
predict.pyで0と1の画像を含んだtestデータに対して、異常度を計算し、aucrocかaccuracyのスコアを表示]

!python3 train.py -c config.json
!python3 predict.py -c config.json
で実行できる

聞きたいこと
学習結果とかの何をどう表示させるか、こんな感じでいいのか、

まだできていないこと
自分のpytorchのコードに一回落とし込んでそのまま作ったので、lightningにはまだできていない

もとのcolabのコード
https://colab.research.google.com/drive/1K4DM2H8DyVKx9j05k52qLi_Ze2pyWD6y#scrollTo=-jMfIhJC_5Gc