# 災害支援優先度・参考指標アプリ（デモ）

## ⚠️ 注意事項
- デモ・学習目的のアプリです
- 実際の災害対応には使用できません
- 表示されるデータはすべて架空のサンプルです

---

## 📋 アプリ概要
- 複数のベイズ統計手法を統合して、災害支援要請の優先度を「参考指標」として可視化
- 使用手法:
  - 共役事前分布（解析的事後分布）
  - MCMCサンプリング（PyMC）
  - ベイジアンネットワーク（pgmpy）
  - ナイーブベイズ（scikit-learn）

詳細な技術情報は [TECHNICAL.md](./TECHNICAL.md) を参照してください。

---

## 🚀 クイックスタート

```bash
# 1. 推奨 Python バージョン（pyenv 利用可）
# Python 3.11 または 3.12 推奨
pyenv install 3.12.2       # インストール
pyenv local 3.12.2         # プロジェクト内のみ有効化

# 2. リポジトリのクローン
git clone <repository-url>
cd bayesian-disaster-priority

# 3. 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 4. 依存パッケージのインストール
pip install --upgrade pip
pip install -r requirements.txt

# 5. アプリ起動
streamlit run app.py

# ⚠️ 注意
# Graphviz 本体がない場合、BN 可視化はスキップされますが、
# アプリ本体は正常に動作します。
