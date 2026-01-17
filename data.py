"""
災害支援要請データ管理モジュール（デモ・学習用）

注意：このモジュールは架空データを提供するものです。
実運用を想定したものではなく、アプリのデモンストレーションや
学習目的でのみ使用してください。

実際の災害支援システムでは、データベースやAPIから
リアルタイムのデータを取得する必要があります。
"""

from typing import Any, Dict, List, Optional


# 支援要請データの型定義
# 各要請は辞書型で、以下のキーを持つ
SupportRequest = Dict[str, Any]


# 架空の支援要請データ
# 注意：これはデモ・学習用のサンプルデータです
_SAMPLE_REQUESTS: List[SupportRequest] = [
    {
        "id": 1,
        "title": "高齢者世帯・停電3日目",
        "description": "80代夫婦のみの世帯。停電が3日間続いており、冷蔵庫内の食料が心配。近隣からの報告によると、過去に同様の状況で支援が必要だった。",
        "successes": 15,
        "total_trials": 20,
        "urgency_weight": 0.9
    },
    {
        "id": 2,
        "title": "孤立集落・道路寸断",
        "description": "山間部の集落で道路が寸断され、外部との連絡が取れない状態。数時間前に初めて報告があったばかりで、データは少ないが状況は深刻。",
        "successes": 1,
        "total_trials": 3,
        "urgency_weight": 0.95
    },
    {
        "id": 3,
        "title": "避難所・物資不足の可能性",
        "description": "地域の避難所から定期的に報告が上がっている。過去のデータでは支援が必要なケースが多いが、今回はまだ緊急度は中程度と判断されている。",
        "successes": 25,
        "total_trials": 30,
        "urgency_weight": 0.2
    },
    {
        "id": 4,
        "title": "医療機関・発電機故障",
        "description": "地域の診療所で発電機が故障。過去の類似事例では支援が必要だったケースと不要だったケースが半々程度。緊急度は中程度。",
        "successes": 5,
        "total_trials": 10,
        "urgency_weight": 0.5
    },
    {
        "id": 5,
        "title": "子育て世帯・断水2日目",
        "description": "乳幼児を含む家族。断水が2日間続いている。過去のデータでは、このようなケースで支援が必要だった割合は比較的高い。",
        "successes": 12,
        "total_trials": 15,
        "urgency_weight": 0.75
    }
]


def get_all_requests() -> List[SupportRequest]:
    """
    全支援要請データを返す
    
    注意：この関数は架空のサンプルデータを返します。
    実運用を想定したものではなく、デモ・学習目的でのみ使用してください。
    
    Returns
    -------
    List[SupportRequest]
        全支援要請データのリスト
        各要素は以下のキーを持つ辞書：
        - id: int（要請ID）
        - title: str（要請タイトル）
        - description: str（状況説明）
        - successes: int（過去に支援が必要だった回数）
        - total_trials: int（報告・観測回数）
        - urgency_weight: float（緊急度、0.0〜1.0）
    
    Examples
    --------
    >>> requests = get_all_requests()
    >>> print(f"支援要請数: {len(requests)}")
    >>> for req in requests:
    ...     print(f"ID {req['id']}: {req['title']}")
    """
    # データのコピーを返す（元データの変更を防ぐため）
    return [request.copy() for request in _SAMPLE_REQUESTS]


def get_request_by_id(request_id: int) -> Optional[SupportRequest]:
    """
    IDに対応する支援要請データを返す
    
    注意：この関数は架空のサンプルデータから検索します。
    実運用を想定したものではなく、デモ・学習目的でのみ使用してください。
    
    Parameters
    ----------
    request_id : int
        検索する支援要請のID
    
    Returns
    -------
    Optional[SupportRequest]
        該当する支援要請データ（辞書型）
        存在しない場合は None を返す
        各要素は以下のキーを持つ：
        - id: int（要請ID）
        - title: str（要請タイトル）
        - description: str（状況説明）
        - successes: int（過去に支援が必要だった回数）
        - total_trials: int（報告・観測回数）
        - urgency_weight: float（緊急度、0.0〜1.0）
    
    Examples
    --------
    >>> request = get_request_by_id(1)
    >>> if request:
    ...     print(f"タイトル: {request['title']}")
    ...     print(f"緊急度: {request['urgency_weight']}")
    >>> 
    >>> # 存在しないIDの場合
    >>> request = get_request_by_id(999)
    >>> print(request)  # None
    """
    # IDで該当する要請を検索
    for request in _SAMPLE_REQUESTS:
        if request["id"] == request_id:
            # データのコピーを返す（元データの変更を防ぐため）
            return request.copy()
    
    # 該当するIDが見つからない場合
    return None


if __name__ == "__main__":
    # 使用例
    print("=== 支援要請データ管理モジュールの例 ===\n")
    print("注意：これは架空のサンプルデータです。\n")
    
    # 全データを取得
    print("【全支援要請データ】")
    all_requests = get_all_requests()
    print(f"登録されている要請数: {len(all_requests)}件\n")
    
    for req in all_requests:
        print(f"ID {req['id']}: {req['title']}")
        print(f"  説明: {req['description'][:50]}...")
        print(f"  過去データ: {req['successes']}/{req['total_trials']}回が支援必要")
        print(f"  緊急度: {req['urgency_weight']:.2f}")
        print()
    
    # IDで検索
    print("【ID検索の例】")
    request = get_request_by_id(2)
    if request:
        print(f"ID 2 の要請:")
        print(f"  タイトル: {request['title']}")
        print(f"  説明: {request['description']}")
        print(f"  過去データ: {request['successes']}/{request['total_trials']}回")
        print(f"  緊急度: {request['urgency_weight']:.2f}")
    else:
        print("ID 2 の要請は見つかりませんでした")
    
    print()
    
    # 存在しないIDで検索
    request = get_request_by_id(999)
    if request:
        print(f"ID 999 の要請が見つかりました")
    else:
        print("ID 999 の要請は見つかりませんでした（想定通り）")
