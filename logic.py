"""
災害支援優先度スコア計算モジュール

このモジュールは、複数のベイズ手法を統合して
支援要請の優先度を「参考指標」として計算します。

統合される手法：
1. ベイズ統計（共役事前分布）：事後平均とHDI
2. MCMCサンプリング：複雑な事後分布の近似
3. ベイジアンネットワーク：条件付き確率による推論
4. ナイーブベイズ：テキスト・カテゴリデータからの補助予測

注意：このスコアは意思決定を自動化するものではなく、
あくまで支援判断の参考情報として使用してください。
"""

from typing import Tuple, Dict, Any
from bayes import calculate_posterior_stats, sample_posterior
from bn_model import predict_priority as predict_bn
from nb_model import predict_nb


def calculate_priority_score(
    successes: int,
    total_trials: int,
    urgency_weight: float,
    alpha: float = 1.0,
    beta: float = 1.0
) -> Tuple[float, float, Tuple[float, float]]:
    """
    支援要請の優先度スコア（0〜100）を計算する（旧版、後方互換性のため維持）
    
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


def calculate_priority_score_advanced(
    request_data: Dict[str, Any],
    alpha: float = 1.0,
    beta: float = 1.0,
    use_mcmc: bool = True
) -> Dict[str, Any]:
    """
    複数のベイズ手法を統合して優先度スコアを計算する（拡張版）
    
    技術的説明：
    - ベイズ統計（共役事前分布）：解析的な事後平均とHDI
    - MCMCサンプリング：複雑な事後分布（過去被害規模・人口密度を考慮）の近似
    - ベイジアンネットワーク：条件付き確率による推論
    - ナイーブベイズ：テキスト・カテゴリデータからの補助予測
    
    これらのスコアを重み付け平均で統合：
    priority_score = 0.4 * bayes_result + 0.3 * bn_score + 0.3 * nb_score
    
    Parameters
    ----------
    request_data : Dict[str, Any]
        支援要請データ。以下のキーを含む：
        - successes: int（過去に支援が必要だった回数）
        - total_trials: int（報告・観測回数）
        - urgency_weight: float（緊急度、0.0〜1.0）
        - region: str（地域名）
        - disaster_type: str（災害種類）
        - population_density: float（人口密度、0.0〜1.0）
        - past_severity: float（過去被害規模、0.0〜1.0）
        - description: str（状況説明）
    alpha : float, default=1.0
        ベータ分布の事前分布パラメータα（> 0）
    beta : float, default=1.0
        ベータ分布の事前分布パラメータβ（> 0）
    use_mcmc : bool, default=True
        MCMCサンプリングを使用するかどうか
    
    Returns
    -------
    Dict[str, Any]
        以下のキーを持つ辞書：
        - priority_score: float（統合優先度スコア、0.0〜100.0）
        - bayes_posterior_mean: float（ベイズ統計の事後平均、0.0〜1.0）
        - bayes_hdi_lower: float（94%HDI下限）
        - bayes_hdi_upper: float（94%HDI上限）
        - mcmc_posterior_mean: float（MCMCの事後平均、0.0〜1.0、use_mcmc=Trueの場合）
        - mcmc_hdi_lower: float（MCMCの94%HDI下限）
        - mcmc_hdi_upper: float（MCMCの94%HDI上限）
        - mcmc_samples: np.ndarray（MCMCサンプル配列）
        - bn_score: float（ベイジアンネットワーク推論スコア、0.0〜1.0）
        - nb_score: float（ナイーブベイズ補助スコア、0.0〜1.0）
    
    Examples
    --------
    >>> request_data = {
    ...     "successes": 15,
    ...     "total_trials": 20,
    ...     "urgency_weight": 0.9,
    ...     "region": "都市部",
    ...     "disaster_type": "停電",
    ...     "population_density": 0.85,
    ...     "past_severity": 0.7,
    ...     "description": "80代夫婦のみの世帯。停電が3日間続いている。"
    ... }
    >>> result = calculate_priority_score_advanced(request_data)
    >>> print(f"統合優先度スコア: {result['priority_score']:.2f}")
    >>> print(f"BN推論スコア: {result['bn_score']:.4f}")
    >>> print(f"NB補助スコア: {result['nb_score']:.4f}")
    """
    # 入力値の検証
    successes = request_data.get("successes", 0)
    total_trials = request_data.get("total_trials", 1)
    urgency_weight = request_data.get("urgency_weight", 0.5)
    
    if successes < 0:
        raise ValueError("successes は 0 以上である必要があります")
    if total_trials <= 0:
        raise ValueError("total_trials は 0 より大きい必要があります")
    if successes > total_trials:
        raise ValueError("successes は total_trials 以下である必要があります")
    if not 0.0 <= urgency_weight <= 1.0:
        raise ValueError("urgency_weight は 0.0 〜 1.0 の範囲である必要があります")
    
    # 1. ベイズ統計（共役事前分布）：解析的な事後平均とHDI
    bayes_posterior_mean, bayes_hdi = calculate_posterior_stats(
        alpha=alpha,
        beta=beta,
        successes=successes,
        total_trials=total_trials
    )
    
    # 2. MCMCサンプリング（複雑な事後分布の近似）
    mcmc_result = None
    if use_mcmc:
        try:
            mcmc_result = sample_posterior(
                request_data=request_data,
                alpha=alpha,
                beta=beta,
                draws=2000
            )
        except Exception as e:
            # MCMCが失敗した場合、ベイズ統計の結果を使用
            mcmc_result = {
                "posterior_mean": bayes_posterior_mean,
                "hdi_lower": bayes_hdi[0],
                "hdi_upper": bayes_hdi[1],
                "samples": None
            }
    else:
        mcmc_result = {
            "posterior_mean": bayes_posterior_mean,
            "hdi_lower": bayes_hdi[0],
            "hdi_upper": bayes_hdi[1],
            "samples": None
        }
    
    # 3. ベイジアンネットワーク推論
    try:
        bn_score = predict_bn(request_data)
    except Exception as e:
        # BNが失敗した場合、緊急度をそのまま使用
        bn_score = urgency_weight
    
    # 4. ナイーブベイズ補助予測
    description = request_data.get("description", "")
    disaster_type = request_data.get("disaster_type", "不明")
    try:
        nb_score = predict_nb(description, disaster_type)
    except Exception as e:
        # NBが失敗した場合、緊急度をそのまま使用
        nb_score = urgency_weight
    
    # 5. スコアを統合（重み付け平均）
    # ベイズ統計（MCMC結果）: 40%
    # ベイジアンネットワーク: 30%
    # ナイーブベイズ: 30%
    WEIGHT_BAYES = 0.4
    WEIGHT_BN = 0.3
    WEIGHT_NB = 0.3
    
    combined_score = (
        mcmc_result["posterior_mean"] * WEIGHT_BAYES +
        bn_score * WEIGHT_BN +
        nb_score * WEIGHT_NB
    )
    
    # 0〜100のスコアに変換
    priority_score = combined_score * 100.0
    priority_score = max(0.0, min(100.0, priority_score))
    
    # 結果をまとめる
    result = {
        "priority_score": priority_score,
        "bayes_posterior_mean": bayes_posterior_mean,
        "bayes_hdi_lower": bayes_hdi[0],
        "bayes_hdi_upper": bayes_hdi[1],
        "mcmc_posterior_mean": mcmc_result["posterior_mean"],
        "mcmc_hdi_lower": mcmc_result["hdi_lower"],
        "mcmc_hdi_upper": mcmc_result["hdi_upper"],
        "mcmc_samples": mcmc_result["samples"],
        "bn_score": bn_score,
        "nb_score": nb_score
    }
    
    return result


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
