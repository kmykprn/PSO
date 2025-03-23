## はじめに
- Particle Swarm Optimization(PSO)の実装。
- PSOは、（組み合わせ）最適化問題を解決するためのアルゴリズムの一つ
- 3次元以上の入力にも対応。

## 前提条件
[poetry](https://python-poetry.org/docs/)のインストールが必要です。

## インストール
```
git clone https://github.com/kmykprn/PSO.git
cd PSO
poetry install
```

## 実行
```
poetry run python main.py
```

- デフォルトでは3次元の入力を最適化するPSOが実行されます。
- デフォルトの目的関数は、球体の関数であり、(x, y, z) = (0, 0, 0) のとき f(x, y, z) = 0となります。

実行結果の例
```
Global Best Position: [-0.00120555  0.00422565  0.00078391]
Global Best Value: 1.9924028516587042e-05
```
- Global Best Position は、PSOによる探索の結果、最も良い位置（例. X, Y, Z座標）です。(0, 0, 0)に近くなるはずです。
- Global Best Value は、Global Best Positionにおける、目的関数の評価値です。0に近い値ほど良い結果です。


## 使い方
### n次元の入力を受け付ける場合：
- main.pyの`DIMENSION = 3`部分を変更します。
- 例えば2次元の入力を受け付けるときは、`DIMENSION = 2`に設定し、実行します。
- ただし、入力の次元数を減らすと、目的関数も変更となる場合が多いため、後述のように目的関数を適したものに設定します。

### 目的関数の設定
- objective_function.pyの修正
    - objective_function.pyの中に、目的関数を記載します。
- main.pyの修正
    - `from objective_function import ...`部分に、目的関数を記載します。
    - `pso = PSO(objective_function=...`部分に、importした目的関数を設定します。


## 参考にさせて頂いたサイト
- [粒子群最適化(PSO:Particle Swarm Optimization)](https://qiita.com/opticont/items/04a5b4ff41483966987f)
- [最適化アルゴリズムを評価するベンチマーク関数まとめ](https://qiita.com/tomitomi3/items/d4318bf7afbc1c835dda)
- [Particle Swarm Optimization—群れでの探索—](https://www.jstage.jst.go.jp/article/sicejl/47/6/47_459/_pdf)
