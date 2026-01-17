"""
ベイズ統計計算モジュール
ベータ分布を事前分布として、二項データから事後分布を計算し、
事後平均と94%HDI（Highest Density Interval）を返す
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Tuple


def calculate_posterior_stats(
    alpha: float,
    beta: float,
    successes: int,
    total_trials: int,
    credible_interval: float = 0.94
) -> Tuple[float, Tuple[float, float]]:
    """
    ベータ分布を事前分布として、二項データから事後分布を計算し、
    事後平均とHDI（Highest Density Interval）を返す
    
    Parameters
    ----------
    alpha : float
        事前分布（ベータ分布）のαパラメータ（> 0）
    beta : float
        事前分布（ベータ分布）のβパラメータ（> 0）
    successes : int
        成功回数（0以上、total_trials以下）
    total_trials : int
        総試行回数（> 0）
    credible_interval : float, default=0.94
        信頼区間の確率（0 < credible_interval < 1）
    
    Returns
    -------
    posterior_mean : float
        事後平均
    hdi : Tuple[float, float]
        94%HDI（下限、上限）
    
    Examples
    --------
    >>> mean, hdi = calculate_posterior_stats(alpha=1, beta=1, successes=5, total_trials=10)
    >>> print(f"事後平均: {mean:.4f}")
    >>> print(f"94%HDI: [{hdi[0]:.4f}, {hdi[1]:.4f}]")
    """
    # 入力値の検証
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha と beta は正の値である必要があります")
    if successes < 0 or successes > total_trials:
        raise ValueError("successes は 0 以上 total_trials 以下である必要があります")
    if total_trials <= 0:
        raise ValueError("total_trials は正の値である必要があります")
    if not 0 < credible_interval < 1:
        raise ValueError("credible_interval は 0 と 1 の間である必要があります")
    
    # 失敗回数を計算
    failures = total_trials - successes
    
    # 事後分布のパラメータ
    posterior_alpha = alpha + successes
    posterior_beta = beta + failures
    
    # 事後分布オブジェクト
    posterior_dist = stats.beta(posterior_alpha, posterior_beta)
    
    # 事後平均を計算
    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    
    # 94%HDIを計算
    hdi = calculate_hdi(posterior_dist, credible_interval)
    
    return posterior_mean, hdi


def calculate_hdi(distribution: stats.rv_continuous, credible_interval: float) -> Tuple[float, float]:
    """
    連続分布のHDI（Highest Density Interval）を計算
    
    Parameters
    ----------
    distribution : scipy.stats.rv_continuous
        確率分布オブジェクト
    credible_interval : float
        信頼区間の確率（0 < credible_interval < 1）
    
    Returns
    -------
    hdi : Tuple[float, float]
        HDI（下限、上限）
    """
    # 等尾区間を初期値として使用
    lower_tail = (1 - credible_interval) / 2
    upper_tail = 1 - lower_tail
    
    # 初期区間
    initial_lower = distribution.ppf(lower_tail)
    initial_upper = distribution.ppf(upper_tail)
    
    # 区間の幅を最小化する関数
    def interval_width(lower):
        upper = distribution.ppf(distribution.cdf(lower) + credible_interval)
        if upper > 1.0:  # ベータ分布の定義域は[0, 1]
            return np.inf
        return upper - lower
    
    # 最適化：区間の幅を最小化
    result = minimize_scalar(
        interval_width,
        bounds=(0, distribution.ppf(1 - credible_interval)),
        method='bounded'
    )
    
    optimal_lower = result.x
    optimal_upper = distribution.ppf(distribution.cdf(optimal_lower) + credible_interval)
    
    # 境界チェック
    optimal_lower = max(0.0, optimal_lower)
    optimal_upper = min(1.0, optimal_upper)
    
    return (optimal_lower, optimal_upper)


if __name__ == "__main__":
    # 使用例
    print("=== ベイズ統計計算の例 ===\n")
    
    # 例1: 一様事前分布（α=1, β=1）
    print("例1: 一様事前分布（α=1, β=1）")
    print("データ: 10回中5回成功")
    mean1, hdi1 = calculate_posterior_stats(alpha=1, beta=1, successes=5, total_trials=10)
    print(f"事後平均: {mean1:.4f}")
    print(f"94%HDI: [{hdi1[0]:.4f}, {hdi1[1]:.4f}]\n")
    
    # 例2: 情報量の多い事前分布
    print("例2: 情報量の多い事前分布（α=5, β=5）")
    print("データ: 20回中15回成功")
    mean2, hdi2 = calculate_posterior_stats(alpha=5, beta=5, successes=15, total_trials=20)
    print(f"事後平均: {mean2:.4f}")
    print(f"94%HDI: [{hdi2[0]:.4f}, {hdi2[1]:.4f}]\n")
    
    # 例3: 非対称な事前分布
    print("例3: 非対称な事前分布（α=2, β=8）")
    print("データ: 30回中10回成功")
    mean3, hdi3 = calculate_posterior_stats(alpha=2, beta=8, successes=10, total_trials=30)
    print(f"事後平均: {mean3:.4f}")
    print(f"94%HDI: [{hdi3[0]:.4f}, {hdi3[1]:.4f}]")
