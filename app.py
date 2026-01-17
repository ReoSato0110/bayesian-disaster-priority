"""
災害支援優先度・参考指標デモアプリ

このアプリは、過去のデータと緊急度から支援要請の優先度を
「参考指標」として可視化します。

注意：このアプリは意思決定を自動化するものではありません。
あくまで判断の補助情報を提供するデモアプリです。
"""

import streamlit as st
from data import get_all_requests
from logic import calculate_priority_score

# ページ設定
st.set_page_config(
    page_title="災害支援優先度・参考指標",
    layout="wide",
    page_icon="⚠️"
)

# ============================================
# アプリ全体の説明
# ============================================
st.title("⚠️ 災害支援優先度・参考指標（デモアプリ）")

# 重要な注意書きを目立つように表示
st.warning("""
**⚠️ 重要な注意事項**

このアプリは**参考指標**を提供するデモアプリです。
- 意思決定を自動化するものではありません
- 実際の判断は専門家や現場の状況を総合的に考慮して行ってください
- 表示されているデータは**架空のサンプルデータ**です
""")

st.info("""
このアプリでは、過去のデータと緊急度から支援要請の優先度を数値化します。
支援要請を選択すると、自動的に優先度スコアが計算・表示されます。
""")

# ============================================
# 支援要請の選択UI
# ============================================
st.header("📋 支援要請の選択")

# 全支援要請データを取得
all_requests = get_all_requests()

# selectbox用の選択肢を作成（"ID: タイトル"形式）
request_options = {f"ID {req['id']}: {req['title']}": req['id'] for req in all_requests}
request_labels = list(request_options.keys())

# 支援要請を選択
selected_label = st.selectbox(
    "支援要請を選択してください",
    options=request_labels,
    index=0,  # デフォルトで最初の要請を選択
    help="リストから支援要請を選択すると、自動的に優先度スコアが計算されます"
)

# 選択された要請のIDを取得
selected_request_id = request_options[selected_label]

# 選択された要請のデータを取得
selected_request = next(req for req in all_requests if req['id'] == selected_request_id)

# ============================================
# 選択された支援要請の詳細表示
# ============================================
st.header("📄 支援要請の詳細")

# タイトル
st.subheader(selected_request['title'])

# 説明
st.write("**状況説明**")
st.write(selected_request['description'])

# データを2カラムで表示
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="過去データ",
        value=f"{selected_request['successes']} / {selected_request['total_trials']} 回",
        help="過去に支援が必要だった回数 / 報告・観測回数"
    )

with col2:
    st.metric(
        label="緊急度",
        value=f"{selected_request['urgency_weight']:.2f}",
        help="主観的な緊急度（0.0 = 低い、1.0 = 非常に高い）"
    )

# 緊急度を視覚的に表示
st.write("**緊急度の視覚化**")
st.progress(
    selected_request['urgency_weight'],
    text=f"緊急度: {selected_request['urgency_weight']*100:.0f}%"
)

# ============================================
# 優先度スコア計算と表示
# ============================================
st.header("📊 優先度スコア（参考指標）")

# スコアを計算
try:
    priority_score, posterior_mean, hdi = calculate_priority_score(
        successes=selected_request['successes'],
        total_trials=selected_request['total_trials'],
        urgency_weight=selected_request['urgency_weight'],
        alpha=1.0,  # 一様事前分布
        beta=1.0    # 一様事前分布
    )
    
    # メインのスコア表示（大きく目立つように）
    st.metric(
        label="優先度スコア（参考指標）",
        value=f"{priority_score:.1f}",
        help="0〜100のスコア。値が大きいほど優先度が高いことを示す参考指標です。"
    )
    
    # スコアを視覚的に表示
    st.progress(
        priority_score / 100.0,
        text=f"優先度スコア: {priority_score:.1f} / 100"
    )
    
    # 詳細情報を展開可能なセクションに
    with st.expander("📈 詳細な統計情報", expanded=False):
        # 事後平均
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="事後平均（確率）",
                value=f"{posterior_mean:.4f}",
                help="過去データから推定される支援必要確率"
            )
        
        with col2:
            st.metric(
                label="94%HDI（下限）",
                value=f"{hdi[0]:.4f}",
                help="94%最高密度区間の下限"
            )
        
        st.metric(
            label="94%HDI（上限）",
            value=f"{hdi[1]:.4f}",
            help="94%最高密度区間の上限"
        )
        
        # HDI区間の説明
        st.caption("""
        **94%HDI（Highest Density Interval）について**  
        事後分布の94%最高密度区間です。真の確率がこの区間内にある可能性が94%であることを示します。
        """)
    
    # 注意書き
    st.caption("""
    ⚠️ このスコアは参考指標です。実際の支援判断は、専門家の意見や現場の状況を
    総合的に考慮して行ってください。
    """)

except Exception as e:
    st.error(f"スコア計算中にエラーが発生しました: {str(e)}")
    st.exception(e)

# ============================================
# フッター
# ============================================
st.divider()
st.caption("""
**このアプリについて**  
- このアプリはデモ・学習目的で作成されています
- 表示されているデータは架空のサンプルデータです
- 実運用を想定したものではありません
""")
