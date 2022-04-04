# DeepLearning_Basic_Classification
名前：犬猫判定ツール
機能：Dog_Cat_tester.pyで画像のパスを選ぶことでそれが犬なのか猫なのか判定することができる。

1.Animal_scrayper.pyでまずは犬猫の画像を収集
2.Dog_Cat_ClassificationModel.pyで犬猫の画像を使って、
　VGG16のファインチューニングを実施。
  学習した重みを出力する。(.gitignoreに重みを記載)
3.Dog_Cat_tester.pyでモデルに重みを読み込ませ、何らかの画像を入力情報として選択する
4.最後にDog_Cat_test.pyを実行し、犬か猫かの判定を行う

