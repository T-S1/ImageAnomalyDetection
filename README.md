# ImageAnomalyDetection

本レポジトリは，画像に基づく異常検知の基礎を学ぶことを目的とする．  
想定するケースは，製品の欠陥の自動検出とする．
概要を以下に示す．
- 差分検出に基づく異常検知
- シンプルな深層学習モデルによる異常検知

ただし，画像データは予め用意されているものとする．

## Requirements

- tensorflow 2.x
- opencv-python
- numpy
- scikit-learn
- tqdm
- matplotlib
- pydot
- Graphvis

## Note

Windowsでは https://graphviz.org/download/ からGraphvisをダウンロード，環境変数のPathに加え，再起動する必要があります．