# 災害支援優先度・参考指標アプリ（デモ）

## ⚠️ 注意事項
- デモ・学習目的のアプリです。
- 実際の災害対応には使用できません。
- 表示データはすべて架空のサンプルです。

---

## 📋 アプリ概要
- 複数のベイズ統計手法を統合して、災害支援要請の優先度を「参考指標」として可視化
- 使用手法:
  - 共役事前分布による解析的事後分布
  - MCMCサンプリング（PyMC）
  - ベイジアンネットワーク（pgmpy）
  - ナイーブベイズ（scikit-learn）

---

## 🚀 クイックスタート
```bash
git clone <repository-url>
cd bayesian-disaster-priority

# 仮想環境作成・有効化
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 依存パッケージインストール
pip install -r requirements.txt

# アプリ起動
streamlit run app.py

Graphviz 本体がない場合、BN 可視化はスキップされます。アプリ本体は正常に動作します。

---

## 📦 依存関係

| パッケージ                      | 用途                            |
| -------------------------- | ----------------------------- |
| `streamlit`                | Web UI                        |
| `numpy`, `scipy`, `pandas` | 数値計算・データ処理                    |
| `pymc`, `arviz`            | MCMCサンプリング・ベイズ統計              |
| `pgmpy`                    | ベイジアンネットワーク                   |
| `scikit-learn`             | ナイーブベイズ分類                     |
| `altair`                   | 可視化                           |
| `graphviz`                 | BN 可視化（Pythonバインディング、本体は別途必要） |

---

## 📁 プロジェクト構成

```
app.py              # メイン
data.py             # データ管理
bayes.py            # ベイズ計算
logic.py            # 優先度計算
bn_model.py         # ベイジアンネットワーク
nb_model.py         # ナイーブベイズ
requirements.txt
README.md
```