# NumRecognition_tensorflow
数字画像を分類してくれるプログラムです。<br>

## 概要
PythonのTensorFlow（機械学習）とKeras（ニューラルネットワーク）を用いて、<br>
数字画像の機械学習を行い、入力画像の予測結果や正答率と損失率の評価、評価指標を出力します。<br>

## 使い方
1. 訓練用データとテスト用データを用意する。<br>
   (それぞれ「train_data」と「test_data」というディレクトリを作成しておき、<br>
   ファイル名を「train0.png」から枚数分0を増やす。)<br>
2. main.pyのmain関数の最後で、出力したいもののコメントアウトを外す。<br>
3. main.pyを実行する。<br>

## 注意事項
現在のバージョンのプログラムでは、0 と 1 と 2 に対応しています。<br>
また、数字画像はあらかじめ 28 x 28 に編集しています。<br>
<br>
今後のバージョンアップで他の数字にも対応させていきます。<br>

### 製作者
名前：Junta Asaka