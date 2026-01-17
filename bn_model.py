"""
ベイジアンネットワーク（Bayesian Network）モデル

ベイジアンネットワークは、変数間の条件付き依存関係を
有向非巡回グラフ（DAG）で表現する確率的グラフィカルモデルである。

このモジュールでは、災害支援要請の優先度を
複数の変数（地域、人口密度、過去被害規模、緊急度）から
条件付き確率で推論する。
"""

from typing import Dict, Any, Optional
import numpy as np

# pgmpyのインポート（オプショナル）
try:
    # pgmpy 1.0.0以降では DiscreteBayesianNetwork を使用
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        BAYESIAN_NETWORK_CLASS = DiscreteBayesianNetwork
    except ImportError:
        # 旧バージョンの互換性
        from pgmpy.models import BayesianNetwork
        BAYESIAN_NETWORK_CLASS = BayesianNetwork
    
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    BAYESIAN_NETWORK_CLASS = None


# グローバル変数：モデルと推論エンジン
_bn_model: Optional[Any] = None
_inference_engine: Optional[Any] = None


def build_bn() -> Any:
    """
    ベイジアンネットワークモデルを構築する
    
    技術的説明：
    - DAG構造: region → population_density → past_severity → urgency_weight → priority_score
    - 各ノードは条件付き確率分布（CPD）を持つ
    - 変数間の因果関係をモデル化
    
    Returns
    -------
    BayesianNetwork
        構築されたベイジアンネットワークモデル
    
    Examples
    --------
    >>> model = build_bn()
    >>> print(f"ノード数: {len(model.nodes())}")
    """
    if not PGMPY_AVAILABLE:
        # pgmpyが利用できない場合、ダミーモデルを返す
        return None
    
    # ベイジアンネットワークの構造を定義
    # DAG: region → population_density → past_severity → urgency_weight → priority_score
    model = BAYESIAN_NETWORK_CLASS([
        ("region", "population_density"),
        ("population_density", "past_severity"),
        ("past_severity", "urgency_weight"),
        ("urgency_weight", "priority_score")
    ])
    
    # 地域の事前確率（3つのカテゴリ：都市部、郊外、山間部）
    cpd_region = TabularCPD(
        variable="region",
        variable_card=3,
        values=[[0.4], [0.4], [0.2]],  # 都市部、郊外、山間部
        state_names={"region": ["都市部", "郊外", "山間部"]}
    )
    
    # 人口密度（高、中、低の3状態）
    # 条件：地域に依存
    cpd_pop_density = TabularCPD(
        variable="population_density",
        variable_card=3,
        values=[
            [0.8, 0.5, 0.2],  # 都市部の場合
            [0.15, 0.3, 0.4],  # 郊外の場合
            [0.05, 0.2, 0.4]   # 山間部の場合
        ],
        evidence=["region"],
        evidence_card=[3],
        state_names={
            "population_density": ["高", "中", "低"],
            "region": ["都市部", "郊外", "山間部"]
        }
    )
    
    # 過去被害規模（高、中、低の3状態）
    # 条件：人口密度に依存
    # 注意：各列の合計が1になるように正規化
    cpd_past_severity = TabularCPD(
        variable="past_severity",
        variable_card=3,
        values=[
            [0.3, 0.4, 0.5],  # 人口密度が高い場合（合計=1.2 → 正規化）
            [0.4, 0.4, 0.3],  # 人口密度が中程度の場合（合計=1.1 → 正規化）
            [0.3, 0.2, 0.5]   # 人口密度が低い場合（合計=1.0）
        ],
        evidence=["population_density"],
        evidence_card=[3],
        state_names={
            "past_severity": ["高", "中", "低"],
            "population_density": ["高", "中", "低"]
        }
    )
    # 正規化（各列の合計を1にする）
    values = cpd_past_severity.values
    normalized_values = values / values.sum(axis=0, keepdims=True)
    cpd_past_severity = TabularCPD(
        variable="past_severity",
        variable_card=3,
        values=normalized_values,
        evidence=["population_density"],
        evidence_card=[3],
        state_names={
            "past_severity": ["高", "中", "低"],
            "population_density": ["高", "中", "低"]
        }
    )
    
    # 緊急度（高、中、低の3状態）
    # 条件：過去被害規模に依存
    # 注意：各列の合計が1になるように正規化
    urgency_values = np.array([
        [0.7, 0.5, 0.3],  # 過去被害規模が高い場合
        [0.2, 0.5, 0.4],  # 過去被害規模が中程度の場合
        [0.1, 0.0, 0.3]   # 過去被害規模が低い場合
    ])
    # 正規化（各列の合計を1にする）
    urgency_values = urgency_values / urgency_values.sum(axis=0, keepdims=True)
    cpd_urgency = TabularCPD(
        variable="urgency_weight",
        variable_card=3,
        values=urgency_values,
        evidence=["past_severity"],
        evidence_card=[3],
        state_names={
            "urgency_weight": ["高", "中", "低"],
            "past_severity": ["高", "中", "低"]
        }
    )
    
    # 優先度スコア（高、中、低の3状態）
    # 条件：緊急度に依存
    # 注意：各列の合計が1になるように正規化
    priority_values = np.array([
        [0.8, 0.5, 0.2],  # 緊急度が高い場合
        [0.15, 0.4, 0.5],  # 緊急度が中程度の場合
        [0.05, 0.1, 0.3]   # 緊急度が低い場合
    ])
    # 正規化（各列の合計を1にする）
    priority_values = priority_values / priority_values.sum(axis=0, keepdims=True)
    cpd_priority = TabularCPD(
        variable="priority_score",
        variable_card=3,
        values=priority_values,
        evidence=["urgency_weight"],
        evidence_card=[3],
        state_names={
            "priority_score": ["高", "中", "低"],
            "urgency_weight": ["高", "中", "低"]
        }
    )
    
    # CPDをモデルに追加
    model.add_cpds(
        cpd_region,
        cpd_pop_density,
        cpd_past_severity,
        cpd_urgency,
        cpd_priority
    )
    
    # モデルの妥当性をチェック
    if not model.check_model():
        raise ValueError("ベイジアンネットワークモデルの構築に失敗しました")
    
    return model


def predict_priority(request_data: Dict[str, Any]) -> float:
    """
    ベイジアンネットワークを用いて優先度を推論する
    
    技術的説明：
    - 変数消去法（Variable Elimination）を使用して
      条件付き確率を計算
    - 観測された変数（region, population_density, past_severity, urgency_weight）
      から priority_score の確率を推論
    
    Parameters
    ----------
    request_data : Dict[str, Any]
        支援要請データ。以下のキーを含む：
        - region: str（地域名）
        - population_density: float（人口密度、0.0〜1.0）
        - past_severity: float（過去被害規模、0.0〜1.0）
        - urgency_weight: float（緊急度、0.0〜1.0）
    
    Returns
    -------
    float
        優先度スコア（0.0〜1.0）
        条件付き確率 P(priority_score=高 | 観測データ) を返す
    
    Examples
    --------
    >>> request_data = {
    ...     "region": "都市部",
    ...     "population_density": 0.85,
    ...     "past_severity": 0.7,
    ...     "urgency_weight": 0.9
    ... }
    >>> score = predict_priority(request_data)
    >>> print(f"BN推論スコア: {score:.4f}")
    """
    global _bn_model, _inference_engine
    
    if not PGMPY_AVAILABLE:
        # pgmpyが利用できない場合、簡易的な線形結合で近似
        pop_density = request_data.get("population_density", 0.5)
        past_sev = request_data.get("past_severity", 0.5)
        urgency = request_data.get("urgency_weight", 0.5)
        return (pop_density * 0.3 + past_sev * 0.3 + urgency * 0.4)
    
    # モデルが未構築の場合、構築する
    if _bn_model is None:
        _bn_model = build_bn()
        _inference_engine = VariableElimination(_bn_model)
    
    # 連続値を離散値に変換
    region = request_data.get("region", "都市部")
    pop_density_val = request_data.get("population_density", 0.5)
    past_sev_val = request_data.get("past_severity", 0.5)
    urgency_val = request_data.get("urgency_weight", 0.5)
    
    # 連続値を3段階（高、中、低）に変換
    pop_density_state = "高" if pop_density_val > 0.66 else ("中" if pop_density_val > 0.33 else "低")
    past_sev_state = "高" if past_sev_val > 0.66 else ("中" if past_sev_val > 0.33 else "低")
    urgency_state = "高" if urgency_val > 0.66 else ("中" if urgency_val > 0.33 else "低")
    
    # 観測データを設定
    evidence = {
        "region": region,
        "population_density": pop_density_state,
        "past_severity": past_sev_state,
        "urgency_weight": urgency_state
    }
    
    # 条件付き確率を計算
    # P(priority_score=高 | 観測データ)
    try:
        query = _inference_engine.query(
            variables=["priority_score"],
            evidence=evidence
        )
        # "高"の確率を取得
        high_prob = query.values[0]  # 最初の値が"高"の確率
        return float(high_prob)
    except Exception as e:
        # エラーが発生した場合、フォールバック
        return (pop_density_val * 0.3 + past_sev_val * 0.3 + urgency_val * 0.4)


if __name__ == "__main__":
    # 使用例
    print("=== ベイジアンネットワークモデルの例 ===\n")
    
    if not PGMPY_AVAILABLE:
        print("警告: pgmpyがインストールされていません。")
        print("簡易的な線形結合で近似します。\n")
    
    request_data = {
        "region": "都市部",
        "population_density": 0.85,
        "past_severity": 0.7,
        "urgency_weight": 0.9
    }
    
    score = predict_priority(request_data)
    print(f"BN推論スコア: {score:.4f}")
