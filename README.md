# 災害支援優先度・参考指標アプリ（拡張版デモ）

## ⚠️ 重要な注意事項

本アプリは**デモ・学習目的**で作成されたものです。実際の災害対応や意思決定を自動化するものではなく、表示されるデータはすべて架空のサンプルデータです。実運用・実災害への利用は想定していません。

---

## 📋 アプリ概要

複数のベイズ統計手法を統合して、災害支援要請の優先度を「参考指標」として可視化するStreamlitデモアプリです。

### 使用技術

- **ベイズ統計（共役事前分布）**: 過去データから解析的に事後分布を計算
- **MCMCサンプリング**: 複雑な事後分布（過去被害規模・人口密度を考慮）を近似
- **ベイジアンネットワーク**: 変数間の条件付き確率による推論
- **ナイーブベイズ**: テキスト・カテゴリデータからの補助予測

これらの手法を統合し、0〜100の優先度スコアとして表示します。

---

## 🚀 クイックスタート

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd bayesian-disaster-priority
```

### 2. 仮想環境の作成と有効化

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows
```

### 3. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 4. Graphviz本体のインストール（BN可視化用、オプション）

**macOS:**
```bash
brew install graphviz
```

**Linux:**
```bash
sudo apt-get install graphviz  # または sudo yum install graphviz
```

> **注意**: Graphviz本体がなくてもアプリは動作します。BN可視化のみスキップされます。

### 5. アプリの起動

```bash
streamlit run app.py
```

ブラウザが自動的に開き、アプリが表示されます。

---

## 📦 依存関係

| パッケージ | 用途 |
|-----------|------|
| `streamlit` | Webアプリケーションフレームワーク |
| `numpy`, `scipy`, `pandas` | 数値計算・データ処理 |
| `pymc`, `arviz` | MCMCサンプリング・ベイズ統計 |
| `pgmpy` | ベイジアンネットワーク構築・推論 |
| `scikit-learn` | ナイーブベイズ分類器 |
| `altair` | インタラクティブ可視化 |
| `graphviz` | BN可視化（Pythonバインディング、本体は別途必要） |

> **注意**: `graphviz`パッケージはPythonバインディングのみです。Graphviz本体（`dot`コマンド）は別途インストールが必要です。

---

## 🔧 トラブルシューティング

### Graphvizが見つからない

**症状**: 「Graphvizがインストールされていません」と表示される

**解決方法**:
```bash
# Graphviz本体のインストール確認
which dot  # macOS/Linux
dot -V     # バージョン確認

# PATHに追加（必要に応じて）
export PATH="/opt/homebrew/bin:$PATH"  # macOS (Homebrew)
```

> **注意**: Graphvizがなくてもアプリは動作します。BN可視化のみスキップされます。

### pgmpyのインポートエラー

**症状**: 「pgmpyが利用できないため、BN可視化をスキップします」と表示される

**解決方法**:
```bash
pip install --upgrade pgmpy
```

### Streamlit Cloudでの動作

Streamlit CloudではGraphviz本体が利用できないため、BN可視化はスキップされます。その他の機能（ベイズ統計、MCMC、BN推論、NB予測）は正常に動作します。

---

## 📚 技術的詳細

### 優先度スコアの計算

```
priority_score = (
    0.4 × MCMC事後平均 +
    0.3 × BN推論スコア +
    0.3 × NB補助スコア
) × 100
```

### ベイズ統計手法の説明

#### 1. ベイズ統計（共役事前分布）

ベータ分布を事前分布、二項分布を尤度として使用。事後分布は解析的に計算可能。

- **事後平均**: `(α + s) / (α + β + n)`
- **94%HDI**: 最高密度区間を数値最適化で計算

#### 2. MCMCサンプリング

複雑な事後分布（過去被害規模・人口密度を考慮）をPyMCで近似。

- **NUTSサンプラー**: 2000サンプル、バーンイン500
- **用途**: 解析解が得られない複雑なモデルに対応

#### 3. ベイジアンネットワーク

変数間の条件付き依存関係をDAGで表現。

- **構造**: `region → population_density → past_severity → urgency_weight → priority_score`
- **推論**: 変数消去法（Variable Elimination）で条件付き確率を計算

#### 4. ナイーブベイズ

テキスト（description）とカテゴリ（disaster_type）から補助予測。

- **特徴抽出**: Bag-of-Words（1-gram, 2-gram）
- **分類器**: MultinomialNB（ラプラススムージング適用）

---

## 📁 プロジェクト構成

```
bayesian-disaster-priority/
├── app.py              # Streamlitアプリ（メイン）
├── data.py            # 支援要請データ管理
├── bayes.py           # ベイズ統計計算（共役事前分布、MCMC）
├── logic.py           # 優先度スコア計算（統合）
├── bn_model.py        # ベイジアンネットワークモデル
├── nb_model.py        # ナイーブベイズモデル
├── requirements.txt   # 依存パッケージ一覧
└── README.md          # このファイル
```

---

## 📝 ライセンス

本アプリはデモ・学習目的で作成されています。
