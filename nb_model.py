"""
ナイーブベイズ（Naive Bayes）モデル

ナイーブベイズは、特徴量間の独立性を仮定した
ベイズ分類器である。

このモジュールでは、テキストデータ（description）と
カテゴリデータ（disaster_type）から
支援要請の優先度を補助的に予測する。
"""

from typing import Dict, Any, List, Optional
import numpy as np

# scikit-learnのインポート（オプショナル）
try:
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# グローバル変数：モデルと前処理
_nb_model: Optional[Any] = None
_vectorizer: Optional[Any] = None
_label_encoder: Optional[Any] = None
_trained: bool = False


def train_nb_model(request_list: List[Dict[str, Any]]) -> None:
    """
    ナイーブベイズモデルを訓練する
    
    技術的説明：
    - テキストデータ（description）をBag-of-Wordsでベクトル化
    - カテゴリデータ（disaster_type）をラベルエンコーディング
    - ラプラススムージング（alpha=1.0）を適用
    - 対数確率を計算して数値的安定性を確保
    
    Parameters
    ----------
    request_list : List[Dict[str, Any]]
        訓練用の支援要請データリスト
        各要素は以下のキーを含む：
        - description: str（状況説明）
        - disaster_type: str（災害種類）
        - urgency_weight: float（緊急度、0.0〜1.0、ラベルとして使用）
    
    Examples
    --------
    >>> requests = get_all_requests()
    >>> train_nb_model(requests)
    >>> print("モデル訓練完了")
    """
    global _nb_model, _vectorizer, _label_encoder, _trained
    
    if not SKLEARN_AVAILABLE:
        _trained = True
        return
    
    if len(request_list) == 0:
        raise ValueError("訓練データが空です")
    
    # テキストデータとカテゴリデータを抽出
    descriptions = [req.get("description", "") for req in request_list]
    disaster_types = [req.get("disaster_type", "不明") for req in request_list]
    urgency_weights = [req.get("urgency_weight", 0.5) for req in request_list]
    
    # ラベルを3段階に変換（高、中、低）
    labels = []
    for urgency in urgency_weights:
        if urgency > 0.66:
            labels.append("高")
        elif urgency > 0.33:
            labels.append("中")
        else:
            labels.append("低")
    
    # テキストをベクトル化（Bag-of-Words）
    _vectorizer = CountVectorizer(
        max_features=50,  # 最大特徴数
        stop_words=None,  # 日本語のストップワードは使用しない
        ngram_range=(1, 2)  # 1-gramと2-gramを使用
    )
    X_text = _vectorizer.fit_transform(descriptions)
    
    # カテゴリデータをエンコード
    _label_encoder = LabelEncoder()
    X_category = _label_encoder.fit_transform(disaster_types).reshape(-1, 1)
    
    # テキストとカテゴリの特徴量を結合
    from scipy.sparse import hstack
    X = hstack([X_text, X_category])
    
    # ナイーブベイズモデルを訓練
    # alpha=1.0: ラプラススムージング（ゼロ頻度問題を回避）
    _nb_model = MultinomialNB(alpha=1.0)
    _nb_model.fit(X, labels)
    
    _trained = True


def predict_nb(description: str, disaster_type: str = "不明") -> float:
    """
    ナイーブベイズモデルを用いて優先度を予測する
    
    技術的説明：
    - 入力テキストをベクトル化
    - カテゴリデータをエンコード
    - 訓練済みモデルで予測
    - 対数確率から"高"の確率を計算して返す
    
    Parameters
    ----------
    description : str
        支援要請の状況説明（テキスト）
    disaster_type : str, default="不明"
        災害種類（カテゴリ）
    
    Returns
    -------
    float
        補助予測スコア（0.0〜1.0）
        P(優先度=高 | description, disaster_type) を返す
    
    Examples
    --------
    >>> score = predict_nb(
    ...     description="80代夫婦のみの世帯。停電が3日間続いている。",
    ...     disaster_type="停電"
    ... )
    >>> print(f"NB補助スコア: {score:.4f}")
    """
    global _nb_model, _vectorizer, _label_encoder, _trained
    
    if not SKLEARN_AVAILABLE or not _trained:
        # フォールバック：簡易的なテキスト分析
        high_keywords = ["緊急", "深刻", "危険", "孤立", "停電", "断水", "高齢", "乳幼児"]
        low_keywords = ["可能性", "心配", "予備的"]
        
        text_lower = description.lower()
        high_count = sum(1 for kw in high_keywords if kw in text_lower)
        low_count = sum(1 for kw in low_keywords if kw in text_lower)
        
        if high_count > low_count:
            return min(0.8, 0.5 + high_count * 0.1)
        else:
            return max(0.2, 0.5 - low_count * 0.1)
    
    if _nb_model is None or _vectorizer is None or _label_encoder is None:
        raise ValueError("モデルが訓練されていません。train_nb_model()を先に呼び出してください。")
    
    # テキストをベクトル化
    X_text = _vectorizer.transform([description])
    
    # カテゴリデータをエンコード
    try:
        X_category = _label_encoder.transform([disaster_type]).reshape(-1, 1)
    except ValueError:
        # 未知のカテゴリの場合、最初のカテゴリを使用
        X_category = np.array([[0]])
    
    # 特徴量を結合
    from scipy.sparse import hstack
    X = hstack([X_text, X_category])
    
    # 予測（対数確率を取得）
    log_proba = _nb_model.predict_log_proba(X)[0]
    
    # クラス名を取得
    classes = _nb_model.classes_
    
    # "高"の確率を取得
    if "高" in classes:
        high_idx = list(classes).index("高")
        # 対数確率から確率に変換
        prob_high = np.exp(log_proba[high_idx])
        return float(prob_high)
    else:
        # "高"クラスが存在しない場合、最大確率を返す
        return float(np.exp(log_proba.max()))


if __name__ == "__main__":
    # 使用例
    print("=== ナイーブベイズモデルの例 ===\n")
    
    if not SKLEARN_AVAILABLE:
        print("警告: scikit-learnがインストールされていません。")
        print("簡易的なキーワード分析で近似します。\n")
    
    # ダミーデータで訓練
    from data import get_all_requests
    requests = get_all_requests()
    train_nb_model(requests)
    
    # 予測
    description = "80代夫婦のみの世帯。停電が3日間続いており、冷蔵庫内の食料が心配。"
    disaster_type = "停電"
    
    score = predict_nb(description, disaster_type)
    print(f"NB補助スコア: {score:.4f}")
