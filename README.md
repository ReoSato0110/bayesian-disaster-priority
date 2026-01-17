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
推奨 Python バージョン: **3.11–3.12**

```bash
# リポジトリをクローン
git clone <repository-url>
cd bayesian-disaster-priority

# 仮想環境作成・有効化
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# pip 更新 & 依存パッケージインストール
pip install --upgrade pip
pip install -r requirements.txt

# アプリ起動
streamlit run app.py
