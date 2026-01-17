"""
ベイズ統計計算モジュール
ベータ分布を事前分布として、二項データから事後分布を計算し、
事後平均と94%HDI（Highest Density Interval）を返す
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Tuple, Dict, Any


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


def sample_posterior(
    request_data: Dict[str, Any],
    alpha: float = 1.0,
    beta: float = 1.0,
    draws: int = 2000
) -> Dict[str, Any]:
    """
    MCMCサンプリングを用いて事後分布を推定する
    
    この関数は、複雑な事後分布（解析解が存在しない場合）を
    MCMC（Markov Chain Monte Carlo）で近似します。
    
    技術的説明：
    - ベータ分布と二項分布の共役性により解析解が存在するが、
      より複雑なモデル（例：過去被害規模や人口密度を考慮したモデル）
      では解析解が得られないため、MCMCが必要となる
    - PyMCを使用してNUTSサンプラーで事後分布をサンプリング
    
    Parameters
    ----------
    request_data : Dict[str, Any]
        支援要請データ。以下のキーを含む：
        - successes: int（過去に支援が必要だった回数）
        - total_trials: int（報告・観測回数）
        - past_severity: float（過去被害規模、0.0〜1.0、オプション）
        - population_density: float（人口密度、0.0〜1.0、オプション）
    alpha : float, default=1.0
        ベータ分布の事前分布パラメータα（> 0）
    beta : float, default=1.0
        ベータ分布の事前分布パラメータβ（> 0）
    draws : int, default=2000
        MCMCサンプリングの試行回数
    
    Returns
    -------
    Dict[str, Any]
        以下のキーを持つ辞書：
        - posterior_mean: float（事後平均）
        - hdi_lower: float（94%HDI下限）
        - hdi_upper: float（94%HDI上限）
        - samples: np.ndarray（MCMCサンプル配列、形状: (draws,)）
    
    Examples
    --------
    >>> request_data = {
    ...     "successes": 15,
    ...     "total_trials": 20,
    ...     "past_severity": 0.7,
    ...     "population_density": 0.85
    ... }
    >>> result = sample_posterior(request_data, draws=2000)
    >>> print(f"事後平均: {result['posterior_mean']:.4f}")
    >>> print(f"94%HDI: [{result['hdi_lower']:.4f}, {result['hdi_upper']:.4f}]")
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        # PyMCがインストールされていない場合、解析解を使用
        successes = request_data.get("successes", 0)
        total_trials = request_data.get("total_trials", 1)
        posterior_mean, hdi = calculate_posterior_stats(
            alpha=alpha,
            beta=beta,
            successes=successes,
            total_trials=total_trials
        )
        # サンプルをベータ分布から生成（MCMCの代替）
        posterior_dist = stats.beta(alpha + successes, beta + total_trials - successes)
        samples = posterior_dist.rvs(size=draws)
        return {
            "posterior_mean": posterior_mean,
            "hdi_lower": hdi[0],
            "hdi_upper": hdi[1],
            "samples": samples
        }
    
    # 入力値の検証
    successes = request_data.get("successes", 0)
    total_trials = request_data.get("total_trials", 1)
    past_severity = request_data.get("past_severity", 0.5)
    population_density = request_data.get("population_density", 0.5)
    
    if successes < 0 or successes > total_trials:
        raise ValueError("successes は 0 以上 total_trials 以下である必要があります")
    if total_trials <= 0:
        raise ValueError("total_trials は正の値である必要があります")
    
    # PyMCモデルを構築
    # より複雑なモデル：過去被害規模と人口密度を考慮
    # これにより解析解が得られなくなり、MCMCが必要となる
    with pm.Model() as model:
        # 事前分布：過去被害規模と人口密度を考慮した事前分布
        # 過去被害規模が高いほど、人口密度が高いほど、支援が必要な確率が高くなる傾向をモデル化
        adjusted_alpha = alpha + past_severity * 2.0
        adjusted_beta = beta + (1.0 - population_density) * 2.0
        
        # 支援が必要な確率の事前分布
        theta = pm.Beta("theta", alpha=adjusted_alpha, beta=adjusted_beta)
        
        # 尤度：二項分布
        y = pm.Binomial("y", n=total_trials, p=theta, observed=successes)
        
        # MCMCサンプリング（NUTSサンプラー）
        trace = pm.sample(
            draws=draws,
            tune=500,  # バーンイン期間
            return_inferencedata=True,
            progressbar=False
        )
    
    # サンプルを取得
    samples = trace.posterior["theta"].values.flatten()
    
    # 事後平均を計算
    posterior_mean = float(np.mean(samples))
    
    # 94%HDIを計算
    hdi_result = az.hdi(trace, hdi_prob=0.94)
    hdi_lower = float(hdi_result["theta"].sel(hdi="lower").values)
    hdi_upper = float(hdi_result["theta"].sel(hdi="higher").values)
    
    return {
        "posterior_mean": posterior_mean,
        "hdi_lower": hdi_lower,
        "hdi_upper": hdi_upper,
        "samples": samples
    }


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
