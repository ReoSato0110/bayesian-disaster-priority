"""
災害支援優先度・参考指標デモアプリ（拡張版）

このアプリは、複数のベイズ手法を統合して
支援要請の優先度を「参考指標」として可視化します。

使用技術：
1. ベイズ統計（共役事前分布）：解析的な事後平均とHDI
2. MCMCサンプリング：複雑な事後分布の近似（PyMC使用）
3. ベイジアンネットワーク：条件付き確率による推論（pgmpy使用）
4. ナイーブベイズ：テキスト・カテゴリデータからの補助予測（scikit-learn使用）

注意：このアプリは意思決定を自動化するものではありません。
あくまで判断の補助情報を提供するデモアプリです。
"""

import streamlit as st
import pandas as pd
import numpy as np
from data import get_all_requests
from logic import calculate_priority_score, calculate_priority_score_advanced
from nb_model import train_nb_model

# ページ設定
st.set_page_config(
    page_title="災害支援優先度・参考指標",
    layout="wide",
    page_icon="⚠️"
)

# ============================================
# アプリ全体の説明
# ============================================
st.title("⚠️ 災害支援優先度・参考指標（拡張版デモアプリ）")

# 重要な注意書きを目立つように表示
st.warning("""
**⚠️ 重要な注意事項**

このアプリは**参考指標**を提供するデモアプリです。
- 意思決定を自動化するものではありません
- 実際の判断は専門家や現場の状況を総合的に考慮して行ってください
- 表示されているデータは**架空のサンプルデータ**です
""")

st.info("""
このアプリでは、複数のベイズ手法を統合して支援要請の優先度を数値化します。
- **ベイズ統計**: 過去データから事後分布を計算（共役事前分布）
- **MCMCサンプリング**: 複雑な事後分布を近似（過去被害規模・人口密度を考慮）
- **ベイジアンネットワーク**: 条件付き確率による推論
- **ナイーブベイズ**: テキスト・カテゴリデータからの補助予測
""")

# ============================================
# 全支援要請の表表示
# ============================================
st.header("📊 全支援要請一覧")

# 全支援要請データを取得
all_requests = get_all_requests()

# ナイーブベイズモデルを訓練（初回のみ）
if 'nb_trained' not in st.session_state:
    try:
        train_nb_model(all_requests)
        st.session_state.nb_trained = True
    except Exception as e:
        st.warning(f"ナイーブベイズモデルの訓練に失敗しました: {str(e)}")

# 各要請のスコアを計算
table_data = []
for req in all_requests:
    try:
        # 拡張版スコア計算
        result = calculate_priority_score_advanced(req, use_mcmc=True)
        
        table_data.append({
            "ID": req['id'],
            "タイトル": req['title'],
            "地域": req.get('region', '不明'),
            "災害種類": req.get('disaster_type', '不明'),
            "人口密度": f"{req.get('population_density', 0.0):.2f}",
            "過去被害規模": f"{req.get('past_severity', 0.0):.2f}",
            "緊急度": f"{req.get('urgency_weight', 0.0):.2f}",
            "事後平均": f"{result['bayes_posterior_mean']:.2f}",
            "94%HDI": f"[{result['bayes_hdi_lower']:.2f}, {result['bayes_hdi_upper']:.2f}]",
            "BN推論": f"{result['bn_score']:.2f}",
            "NB補助スコア": f"{result['nb_score']:.2f}",
            "優先度スコア": f"{result['priority_score']:.2f}"
        })
    except Exception as e:
        # エラーが発生した場合、基本情報のみ表示
        table_data.append({
            "ID": req['id'],
            "タイトル": req['title'],
            "地域": req.get('region', '不明'),
            "災害種類": req.get('disaster_type', '不明'),
            "人口密度": f"{req.get('population_density', 0.0):.2f}",
            "過去被害規模": f"{req.get('past_severity', 0.0):.2f}",
            "緊急度": f"{req.get('urgency_weight', 0.0):.2f}",
            "事後平均": "計算エラー",
            "94%HDI": "計算エラー",
            "BN推論": "計算エラー",
            "NB補助スコア": "計算エラー",
            "優先度スコア": "計算エラー"
        })

# DataFrameを作成してCursor表として表示
df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================
# 支援要請の選択UI
# ============================================
st.header("📋 支援要請の詳細分析")

# selectbox用の選択肢を作成（"ID: タイトル"形式）
request_options = {f"ID {req['id']}: {req['title']}": req['id'] for req in all_requests}
request_labels = list(request_options.keys())

# 支援要請を選択
selected_label = st.selectbox(
    "詳細を表示する支援要請を選択してください",
    options=request_labels,
    index=0,  # デフォルトで最初の要請を選択
    help="リストから支援要請を選択すると、詳細な分析結果が表示されます"
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

# データをカラムで表示
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="地域",
        value=selected_request.get('region', '不明')
    )

with col2:
    st.metric(
        label="災害種類",
        value=selected_request.get('disaster_type', '不明')
    )

with col3:
    st.metric(
        label="人口密度",
        value=f"{selected_request.get('population_density', 0.0):.2f}"
    )

with col4:
    st.metric(
        label="過去被害規模",
        value=f"{selected_request.get('past_severity', 0.0):.2f}"
    )

col5, col6 = st.columns(2)

with col5:
    st.metric(
        label="過去データ",
        value=f"{selected_request['successes']} / {selected_request['total_trials']} 回",
        help="過去に支援が必要だった回数 / 報告・観測回数"
    )

with col6:
    st.metric(
        label="緊急度",
        value=f"{selected_request['urgency_weight']:.2f}",
        help="主観的な緊急度（0.0 = 低い、1.0 = 非常に高い）"
    )

# ============================================
# 優先度スコア計算と表示（拡張版）
# ============================================
st.header("📊 優先度スコア（参考指標・統合版）")

# スコアを計算
try:
    # 拡張版スコア計算
    result = calculate_priority_score_advanced(selected_request, use_mcmc=True)
    
    # メインのスコア表示（大きく目立つように）
    st.metric(
        label="統合優先度スコア（参考指標）",
        value=f"{result['priority_score']:.1f}",
        help="0〜100のスコア。複数のベイズ手法を統合した参考指標です。"
    )
    
    # スコアを視覚的に表示
    st.progress(
        result['priority_score'] / 100.0,
        text=f"優先度スコア: {result['priority_score']:.1f} / 100"
    )
    
    # 各手法のスコアを表示
    st.subheader("各手法のスコア内訳")
    score_col1, score_col2, score_col3 = st.columns(3)
    
    with score_col1:
        st.metric(
            label="ベイズ統計（MCMC）",
            value=f"{result['mcmc_posterior_mean']:.2f}",
            help="MCMCサンプリングによる事後平均（重み: 40%）"
        )
    
    with score_col2:
        st.metric(
            label="ベイジアンネットワーク",
            value=f"{result['bn_score']:.2f}",
            help="BN推論スコア（重み: 30%）"
        )
    
    with score_col3:
        st.metric(
            label="ナイーブベイズ",
            value=f"{result['nb_score']:.2f}",
            help="NB補助スコア（重み: 30%）"
        )
    
    # ============================================
    # MCMC可視化
    # ============================================
    if result['mcmc_samples'] is not None:
        st.subheader("📈 MCMCサンプリング可視化")
        
        try:
            import altair as alt
            
            # サンプルデータをDataFrameに変換
            samples_df = pd.DataFrame({
                '確率': result['mcmc_samples']
            })
            
            # ヒストグラムを作成
            chart = alt.Chart(samples_df).mark_bar().encode(
                alt.X('確率:Q', bin=alt.Bin(maxbins=50), title='支援必要確率'),
                alt.Y('count()', title='頻度'),
                tooltip=['count()']
            ).properties(
                width=600,
                height=300,
                title='MCMCサンプリング分布（事後分布の近似）'
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # HDI区間を表示
            st.caption(f"""
            **94%HDI**: [{result['mcmc_hdi_lower']:.2f}, {result['mcmc_hdi_upper']:.2f}]  
            この区間は、MCMCサンプリングにより計算された94%最高密度区間です。
            過去被害規模と人口密度を考慮した複雑な事後分布を近似しています。
            """)
        except ImportError:
            st.warning("Altairがインストールされていません。MCMC可視化をスキップします。")
        except Exception as e:
            st.warning(f"MCMC可視化中にエラーが発生しました: {str(e)}")
    
    # ============================================
    # ベイジアンネットワーク可視化
    # ============================================
    st.subheader("🕸️ ベイジアンネットワーク構造")
    
    try:
        from bn_model import build_bn
        import subprocess
        import os
        
        # Graphvizのパスを確認
        graphviz_paths = [
            "/opt/homebrew/bin/dot",
            "/usr/local/bin/dot",
            "/usr/bin/dot"
        ]
        graphviz_found = False
        for path in graphviz_paths:
            if os.path.exists(path):
                # PATHに追加（StreamlitがGraphvizを見つけられるように）
                current_path = os.environ.get("PATH", "")
                if path not in current_path:
                    os.environ["PATH"] = f"{os.path.dirname(path)}:{current_path}"
                graphviz_found = True
                break
        
        # Graphvizが利用可能か確認
        try:
            graphviz_result = subprocess.run(
                ["dot", "-V"],
                capture_output=True,
                text=True,
                timeout=2
            )
            graphviz_available = graphviz_result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            graphviz_available = False
        
        if not graphviz_available:
            st.warning(f"""
            **Graphvizが見つかりません**
            - Graphviz本体のパス: {graphviz_paths[0] if graphviz_found else '見つかりませんでした'}
            - 環境変数PATHにGraphvizのbinディレクトリが含まれているか確認してください
            - 例: `export PATH="/opt/homebrew/bin:$PATH"`
            """)
        
        # BNモデルを構築
        bn_model = build_bn()
        if bn_model is not None:
            # DAGをGraphviz DOT形式の文字列として生成
            dot_lines = ['digraph G {', 'rankdir=LR;', 'node [shape=box];']
            
            # ノードを追加
            for node in bn_model.nodes():
                dot_lines.append(f'  "{node}";')
            
            # エッジを追加
            for edge in bn_model.edges():
                dot_lines.append(f'  "{edge[0]}" -> "{edge[1]}";')
            
            dot_lines.append('}')
            dot_source = '\n'.join(dot_lines)
            
            # Streamlitで表示
            if graphviz_available:
                st.graphviz_chart(dot_source)
            else:
                # Graphvizが利用できない場合、DOTソースを表示
                with st.expander("DOTソースコード（Graphvizが利用できないため）", expanded=False):
                    st.code(dot_source, language="dot")
                st.info("Graphvizが利用できないため、グラフを表示できません。上記のDOTソースをコピーして、オンラインのGraphvizビューアーで表示できます。")
            
            st.caption("""
            **ベイジアンネットワーク構造**  
            DAG: region → population_density → past_severity → urgency_weight → priority_score  
            各ノードは条件付き確率分布（CPD）を持ち、変数間の因果関係をモデル化しています。
            """)
        else:
            st.info("pgmpyが利用できないため、BN可視化をスキップします。")
    except ImportError as e:
        st.warning(f"必要なモジュールがインポートできません: {str(e)}")
        st.info("pgmpyがインストールされているか確認してください: `pip install pgmpy`")
    except Exception as e:
        st.warning(f"BN可視化中にエラーが発生しました: {str(e)}")
        st.exception(e)
    
    # ============================================
    # 詳細な統計情報
    # ============================================
    with st.expander("📈 詳細な統計情報", expanded=False):
        st.write("**ベイズ統計（共役事前分布）**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="事後平均（確率）",
                value=f"{result['bayes_posterior_mean']:.4f}",
                help="過去データから推定される支援必要確率（解析解）"
            )
        
        with col2:
            st.metric(
                label="94%HDI（下限）",
                value=f"{result['bayes_hdi_lower']:.2f}",
                help="94%最高密度区間の下限"
            )
        
        st.metric(
            label="94%HDI（上限）",
            value=f"{result['bayes_hdi_upper']:.2f}",
            help="94%最高密度区間の上限"
        )
        
        st.write("**MCMCサンプリング結果**")
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric(
                label="MCMC事後平均",
                value=f"{result['mcmc_posterior_mean']:.4f}",
                help="MCMCサンプリングによる事後平均（過去被害規模・人口密度を考慮）"
            )
        
        with col4:
            st.metric(
                label="MCMC 94%HDI",
                value=f"[{result['mcmc_hdi_lower']:.2f}, {result['mcmc_hdi_upper']:.2f}]",
                help="MCMCによる94%最高密度区間"
            )
        
        # 技術的説明
        st.caption("""
        **技術的説明**  
        - **ベイズ統計（共役事前分布）**: ベータ分布と二項分布の共役性により解析解が得られる
        - **MCMCサンプリング**: 過去被害規模と人口密度を考慮した複雑な事後分布を近似
        - **ベイジアンネットワーク**: 変数間の条件付き依存関係をDAGで表現
        - **ナイーブベイズ**: テキスト・カテゴリデータから補助的な予測を行う
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
- 使用技術: ベイズ統計、MCMC、ベイジアンネットワーク、ナイーブベイズ
""")
