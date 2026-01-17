"""
災害支援優先度スコア計算モジュール

このモジュールは、過去のデータと主観的な緊急度から、
支援要請の優先度を「参考指標」として計算します。

注意：このスコアは意思決定を自動化するものではなく、
あくまで支援判断の参考情報として使用してください。
"""

from typing import Tuple
from bayes import calculate_posterior_stats


def calculate_priority_score(
    successes: int,
    total_trials: int,
    urgency_weight: float,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Tuple[float, float, Tuple[float, float]]:
    """
    支援要請の優先度スコア（0〜100）を計算する
    
    この関数は、過去のデータ（successes/total_trials）と
    主観的な緊急度（urgency_weight）を組み合わせて、
    支援の優先度を数値化します。
    
    注意：このスコアは参考指標であり、実際の意思決定は
    専門家の判断や現場の状況を総合的に考慮して行ってください。
    
    Parameters
    ----------
    successes : int
        過去に実際に支援が必要だった回数（0以上、total_trials以下）
    total_trials : int
        報告・観測回数（0より大きい）
    urgency_weight : float
        人の主観による緊急度（0.0〜1.0）
        0.0: 緊急度が低い
        1.0: 緊急度が非常に高い
    alpha : float, default=1.0
        ベータ分布の事前分布パラメータα（> 0）
        デフォルトは1.0（一様事前分布）
    beta : float, default=1.0
        ベータ分布の事前分布パラメータβ（> 0）
        デフォルトは1.0（一様事前分布）
    
    Returns
    -------
    priority_score : float
        優先度スコア（0.0〜100.0）
        値が大きいほど優先度が高いことを示す参考指標
    posterior_mean : float
        事後平均（0.0〜1.0）
        過去データから推定される支援必要確率
    hdi : Tuple[float, float]
        94%HDI（下限、上限）
        事後分布の94%最高密度区間
    
    Examples
    --------
    >>> score, mean, hdi = calculate_priority_score(
    ...     successes=5,
    ...     total_trials=10,
    ...     urgency_weight=0.8,
    ...     alpha=1.0,
    ...     beta=1.0
    ... )
    >>> print(f"優先度スコア（参考指標）: {score:.2f}")
    >>> print(f"事後平均: {mean:.4f}")
    >>> print(f"94%HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]")
    """
    # 入力値の検証
    if successes < 0:
        raise ValueError("successes は 0 以上である必要があります")
    if total_trials <= 0:
        raise ValueError("total_trials は 0 より大きい必要があります")
    if successes > total_trials:
        raise ValueError("successes は total_trials 以下である必要があります")
    if not 0.0 <= urgency_weight <= 1.0:
        raise ValueError("urgency_weight は 0.0 〜 1.0 の範囲である必要があります")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha と beta は正の値である必要があります")
    
    # ベイズ統計から事後平均とHDIを計算
    # 過去のデータから、支援が必要だった確率を推定
    posterior_mean, hdi = calculate_posterior_stats(
        alpha=alpha,
        beta=beta,
        successes=successes,
        total_trials=total_trials
    )
    
    # 優先度スコアの計算
    # シンプルな線形結合：事後平均と緊急度の加重平均
    # 事後平均（過去データ）を60%、緊急度（主観）を40%の重みで組み合わせ
    # この重みは調整可能ですが、データ重視の設計としています
    WEIGHT_POSTERIOR = 0.6  # 事後平均の重み
    WEIGHT_URGENCY = 0.4     # 緊急度の重み
    
    # 線形結合：事後平均と緊急度の加重平均
    combined_score = (posterior_mean * WEIGHT_POSTERIOR + 
                      urgency_weight * WEIGHT_URGENCY)
    
    # 0〜100のスコアに変換
    priority_score = combined_score * 100.0
    
    # 念のため範囲チェック（浮動小数点誤差対策）
    priority_score = max(0.0, min(100.0, priority_score))
    
    return priority_score, posterior_mean, hdi


if __name__ == "__main__":
    # 使用例
    print("=== 災害支援優先度スコア計算の例 ===\n")
    print("注意：これらのスコアは参考指標であり、")
    print("実際の意思決定は専門家の判断を優先してください。\n")
    
    # 例1: データが多く、緊急度も高いケース
    print("例1: データが多く、緊急度も高いケース")
    print("  過去データ: 20回中15回が支援必要")
    print("  緊急度: 0.9（非常に高い）")
    score1, mean1, hdi1 = calculate_priority_score(
        successes=15,
        total_trials=20,
        urgency_weight=0.9,
        alpha=1.0,
        beta=1.0
    )
    print(f"  優先度スコア（参考指標）: {score1:.2f}")
    print(f"  事後平均: {mean1:.4f}")
    print(f"  94%HDI: [{hdi1[0]:.4f}, {hdi1[1]:.4f}]\n")
    
    # 例2: データが少ないが、緊急度が高いケース
    print("例2: データが少ないが、緊急度が高いケース")
    print("  過去データ: 3回中1回が支援必要")
    print("  緊急度: 0.95（非常に高い）")
    score2, mean2, hdi2 = calculate_priority_score(
        successes=1,
        total_trials=3,
        urgency_weight=0.95,
        alpha=1.0,
        beta=1.0
    )
    print(f"  優先度スコア（参考指標）: {score2:.2f}")
    print(f"  事後平均: {mean2:.4f}")
    print(f"  94%HDI: [{hdi2[0]:.4f}, {hdi2[1]:.4f}]\n")
    
    # 例3: データは多いが、緊急度が低いケース
    print("例3: データは多いが、緊急度が低いケース")
    print("  過去データ: 30回中25回が支援必要")
    print("  緊急度: 0.2（低い）")
    score3, mean3, hdi3 = calculate_priority_score(
        successes=25,
        total_trials=30,
        urgency_weight=0.2,
        alpha=1.0,
        beta=1.0
    )
    print(f"  優先度スコア（参考指標）: {score3:.2f}")
    print(f"  事後平均: {mean3:.4f}")
    print(f"  94%HDI: [{hdi3[0]:.4f}, {hdi3[1]:.4f}]\n")
    
    # 例4: データも緊急度も中程度のケース
    print("例4: データも緊急度も中程度のケース")
    print("  過去データ: 10回中5回が支援必要")
    print("  緊急度: 0.5（中程度）")
    score4, mean4, hdi4 = calculate_priority_score(
        successes=5,
        total_trials=10,
        urgency_weight=0.5,
        alpha=1.0,
        beta=1.0
    )
    print(f"  優先度スコア（参考指標）: {score4:.2f}")
    print(f"  事後平均: {mean4:.4f}")
    print(f"  94%HDI: [{hdi4[0]:.4f}, {hdi4[1]:.4f}]")
