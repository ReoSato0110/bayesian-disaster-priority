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

### アプリの内部計算・手法の詳細は `TECHNICAL.md` にまとめています。 

<details>
<summary>TECHNICAL.md の詳細を見る</summary>

# 技術仕様書

本ドキュメントは、災害支援優先度・参考指標アプリの技術的な詳細を説明します。理論的背景、計算方法、実装の詳細を扱います。

---

## 1. 優先度スコア計算の詳細

### 1.1 統合計算式

優先度スコアは、3つのベイズ手法の結果を重み付け平均で統合します：

```
priority_score = (
    0.4 × MCMC事後平均 +
    0.3 × BN推論スコア +
    0.3 × NB補助スコア
) × 100
```

最終的なスコアは0〜100の範囲にクリッピングされます。

### 1.2 各手法の影響

#### MCMC事後平均（重み: 40%）

- **役割**: 過去データと追加情報（過去被害規模、人口密度）を統合した事後分布の平均
- **特徴**: 複雑なモデルに対応可能、計算コストが高い
- **影響**: 最も大きな重みを持つため、過去データの傾向を強く反映

#### BN推論スコア（重み: 30%）

- **役割**: 変数間の条件付き依存関係から推論される優先度
- **特徴**: 因果関係をモデル化、観測データが少ない場合でも推論可能
- **影響**: 地域、人口密度、過去被害規模の関係性を反映

#### NB補助スコア（重み: 30%）

- **役割**: テキスト（description）とカテゴリ（disaster_type）からの補助予測
- **特徴**: テキストの意味を簡易的に捉える、計算が高速
- **影響**: 状況説明の内容を反映

### 1.3 PyMC/NUTSサンプリング設定

```python
# bayes.py の sample_posterior() 関数内
trace = pm.sample(
    draws=2000,              # サンプル数
    tune=500,                # バーンイン期間（ウォームアップ）
    return_inferencedata=True,
    progressbar=False
)
```

- **サンプラー**: NUTS (No-U-Turn Sampler)
- **サンプル数**: 2000（デフォルト）
- **バーンイン**: 500（収束までのウォームアップ期間）
- **HDI計算**: ArviZの`az.hdi()`を使用、94%最高密度区間

**計算時間**: 通常1〜3秒程度（環境により異なる）

---

## 2. ベイズ統計の理論

### 2.1 共役事前分布の設定

本アプリでは、**ベータ分布を事前分布、二項分布を尤度**として使用します。

#### 事前分布

```
θ ~ Beta(α, β)
```

- **α, β**: 事前分布のパラメータ（デフォルト: α=1, β=1、一様分布）
- **意味**: 支援が必要な確率θの事前信念

#### 尤度

```
y | θ ~ Binomial(n, θ)
```

- **n**: 総試行回数（total_trials）
- **y**: 成功回数（successes、支援が必要だった回数）

#### 事後分布

共役性により、事後分布もベータ分布になります：

```
θ | y ~ Beta(α + y, β + n - y)
```

### 2.2 事後平均の計算

```
E[θ | y] = (α + y) / (α + β + n)
```

**例**: α=1, β=1, y=15, n=20 の場合
- 事後平均 = (1 + 15) / (1 + 1 + 20) = 16/22 ≈ 0.7273

### 2.3 HDI（最高密度区間）の計算

94%HDIは、事後分布の確率密度が最も高い区間で、94%の確率質量を含む区間です。

**計算方法**:
1. 等尾区間（3%点、97%点）を初期値として使用
2. 区間の幅を最小化する下端点を数値最適化で探索
3. 上端点は累積分布関数（CDF）の逆関数（PPF）で計算

**実装**: `scipy.optimize.minimize_scalar()`を使用

### 2.4 なぜMCMCが必要か

#### 解析解が得られる場合

ベータ分布と二項分布の共役性により、事後分布は解析的に計算可能です。

#### 解析解が困難な場合

より複雑なモデル（例：過去被害規模や人口密度を考慮）では、事後分布が複雑になり、解析解が得られません。

**MCMCモデルの例**:
```python
# 過去被害規模と人口密度を考慮した事前分布
adjusted_alpha = alpha + past_severity * 2.0
adjusted_beta = beta + (1.0 - population_density) * 2.0

theta ~ Beta(adjusted_alpha, adjusted_beta)
y | theta ~ Binomial(n, theta)
```

この場合、事後分布は解析的に計算できず、MCMCサンプリングで近似する必要があります。

---

## 3. ベイジアンネットワーク

### 3.1 DAG構造

```
region → population_density → past_severity → urgency_weight → priority_score
```

**ノード**:
- `region`: 地域（都市部、郊外、山間部）
- `population_density`: 人口密度（高、中、低）
- `past_severity`: 過去被害規模（高、中、低）
- `urgency_weight`: 緊急度（高、中、低）
- `priority_score`: 優先度スコア（高、中、低）

**エッジ**: 親ノードから子ノードへの因果関係を表現

### 3.2 条件付き確率分布（CPD）

各ノードは、親ノードの状態に依存した条件付き確率分布を持ちます。

#### 例: population_density のCPD

```
P(population_density = 高 | region = 都市部) = 0.8
P(population_density = 中 | region = 都市部) = 0.15
P(population_density = 低 | region = 都市部) = 0.05
```

各列（親ノードの状態）の確率の合計は1になります。

### 3.3 推論方法

**変数消去法（Variable Elimination）**を使用して、観測された変数から目的変数の確率を推論します。

**推論の例**:
```python
# 観測データ
evidence = {
    "region": "都市部",
    "population_density": "高",
    "past_severity": "高",
    "urgency_weight": "高"
}

# 推論: P(priority_score = 高 | evidence)
query = inference_engine.query(
    variables=["priority_score"],
    evidence=evidence
)
```

**実装**: pgmpyの`VariableElimination`クラスを使用

---

## 4. ナイーブベイズ

### 4.1 特徴量化

#### Bag-of-Words

テキスト（description）を単語の出現頻度ベクトルに変換します。

**設定**:
- **最大特徴数**: 50
- **N-gram**: 1-gramと2-gram（単語と2連続単語）
- **ストップワード**: 使用しない（日本語対応のため）

**例**:
```
入力: "80代夫婦のみの世帯。停電が3日間続いている。"
出力: [1, 0, 1, 0, 1, ...]  # 各単語の出現有無
```

#### カテゴリエンコード

災害種類（disaster_type）を数値にエンコードします。

**実装**: scikit-learnの`LabelEncoder`を使用

### 4.2 MultinomialNBの使用

```python
_nb_model = MultinomialNB(alpha=1.0)
_nb_model.fit(X, labels)
```

- **alpha=1.0**: ラプラススムージング（加算スムージング）
- **理由**: ゼロ頻度問題を回避（訓練データに出現しなかった単語でも確率が0にならない）

### 4.3 ラプラススムージングの理由

訓練データに出現しなかった単語がテストデータに出現した場合、通常の確率計算では確率が0になり、予測ができません。

**ラプラススムージング**:
```
P(word | class) = (count(word, class) + alpha) / (count(class) + alpha * vocabulary_size)
```

- **alpha=1.0**: 各単語に1回分の「仮想的な出現」を加算
- **効果**: ゼロ頻度問題を回避し、より安定した予測が可能

### 4.4 予測の流れ

1. テキストをBag-of-Wordsでベクトル化
2. カテゴリを数値エンコード
3. 特徴量を結合
4. 訓練済みモデルで対数確率を計算
5. 「高」クラスの確率を返す

---

## 5. 注意事項

### 5.1 デモ・学習目的であること

本アプリは以下の理由により、実運用には適していません：

- **モデルの簡略化**: 実際の災害支援には、より多くの変数と複雑な関係性が必要
- **サンプルデータ**: すべて架空のデータであり、実データではない
- **計算精度**: デモ目的のため、計算精度や検証は限定的

### 5.2 実運用ではない理由

#### モデルの簡略化

- **変数の不足**: 実際には、気象データ、インフラ状況、避難所の収容能力など、多くの変数が必要
- **関係性の単純化**: 変数間の関係は実際にはより複雑で、非線形な相互作用が存在する可能性がある

#### サンプルデータ

- **データの質**: 架空のデータであり、実災害の複雑性を反映していない
- **データ量**: 5件のサンプルデータのみで、統計的な信頼性は低い

#### 計算精度

- **MCMCサンプル数**: 2000サンプルは実運用には不十分な場合がある
- **HDI計算**: 簡易的な実装であり、より厳密な検証が必要
- **重みパラメータ**: 0.4, 0.3, 0.3の重みは固定値であり、データに応じた調整が必要

---

## 6. 参考図と例

### 6.1 ベイジアンネットワークのDAG図

```
┌─────────┐
│ region  │
└────┬────┘
     │
     ▼
┌─────────────────┐
│ population_     │
│ density         │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ past_severity   │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ urgency_weight  │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ priority_score  │
└─────────────────┘
```

### 6.2 計算例

#### 例1: データが多い場合

**入力**:
- successes: 15
- total_trials: 20
- past_severity: 0.7
- population_density: 0.85
- urgency_weight: 0.9

**計算過程**:
1. **ベイズ統計**: 事後平均 = (1 + 15) / (1 + 1 + 20) = 0.7273
2. **MCMC**: 過去被害規模と人口密度を考慮 → 事後平均 ≈ 0.75
3. **BN推論**: 条件付き確率から推論 → スコア ≈ 0.8
4. **NB予測**: テキスト分析 → スコア ≈ 0.7

**統合スコア**:
```
priority_score = (0.75 × 0.4 + 0.8 × 0.3 + 0.7 × 0.3) × 100
                = (0.30 + 0.24 + 0.21) × 100
                = 0.75 × 100
                = 75.0
```

#### 例2: データが少ない場合

**入力**:
- successes: 1
- total_trials: 3
- past_severity: 0.9
- population_density: 0.15
- urgency_weight: 0.95

**計算過程**:
1. **ベイズ統計**: 事後平均 = (1 + 1) / (1 + 1 + 3) = 0.4
2. **MCMC**: 過去被害規模が高いため、事後平均が上昇 → ≈ 0.55
3. **BN推論**: 山間部で過去被害規模が高い → スコア ≈ 0.7
4. **NB予測**: 緊急度が高いテキスト → スコア ≈ 0.85

**統合スコア**:
```
priority_score = (0.55 × 0.4 + 0.7 × 0.3 + 0.85 × 0.3) × 100
                = (0.22 + 0.21 + 0.255) × 100
                = 0.685 × 100
                = 68.5
```

### 6.3 コードスニペット

#### ベイズ統計の計算

```python
from bayes import calculate_posterior_stats

# 事後平均とHDIを計算
posterior_mean, hdi = calculate_posterior_stats(
    alpha=1.0,
    beta=1.0,
    successes=15,
    total_trials=20
)

print(f"事後平均: {posterior_mean:.4f}")
print(f"94%HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]")
```

#### MCMCサンプリング

```python
from bayes import sample_posterior

request_data = {
    "successes": 15,
    "total_trials": 20,
    "past_severity": 0.7,
    "population_density": 0.85
}

result = sample_posterior(request_data, draws=2000)
print(f"MCMC事後平均: {result['posterior_mean']:.4f}")
print(f"94%HDI: [{result['hdi_lower']:.4f}, {result['hdi_upper']:.4f}]")
```

#### ベイジアンネットワーク推論

```python
from bn_model import predict_priority

request_data = {
    "region": "都市部",
    "population_density": 0.85,
    "past_severity": 0.7,
    "urgency_weight": 0.9
}

bn_score = predict_priority(request_data)
print(f"BN推論スコア: {bn_score:.4f}")
```

---

## 参考資料

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
- Koller, D., & Friedman, N. (2009). *Probabilistic Graphical Models*. MIT Press.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.


</details>
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

| ファイル名            | 説明                                |
|----------------------|-----------------------------------|
| `app.py`             | Streamlit アプリのメイン            |
| `data.py`            | 支援要請データ管理                  |
| `bayes.py`           | ベイズ計算（共役事前分布、MCMC）     |
| `logic.py`           | 優先度スコア計算（統合）            |
| `bn_model.py`        | ベイジアンネットワークモデル        |
| `nb_model.py`        | ナイーブベイズモデル                 |
| `requirements.txt`   | 依存パッケージ一覧                  |
| `README.md`          | この README ファイル                 |
| `TECHNICAL.md`       | 内部ロジック・計算式・手法の詳細    |
