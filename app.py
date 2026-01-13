import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import os

# Thư viện tính toán PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 1. CẤU HÌNH TRANG (SIDEBAR COLLAPSED ĐỂ GỌN)
# ==========================================
st.set_page_config(
    page_title="Retail Intelligence Hub",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed" # Mặc định đóng để thấy menu ngang đẹp hơn
)

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_ai = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_qp1q7mct.json")

# ==========================================
# 2. TỪ ĐIỂN ĐA NGÔN NGỮ
# ==========================================
TRANS = {
    "VN": {
        "title": "Trung Tâm Phân Tích Khách Hàng",
        "subtitle": "Hệ thống phân cụm & Khai phá luật kết hợp (Real Data)",
        "tabs": ["Tổng quan & Bộ lọc", "Phân cụm Khách hàng", "Chiến lược (Blog)", "Dữ liệu Chi tiết"],
        
        # Sidebar
        "sb_title": "BỘ LỌC DỮ LIỆU",
        "btn_filter": "Áp Dụng Bộ Lọc",
        "lbl_support": "Độ Phổ Biến (Support)",
        "lbl_conf": "Độ Tin Cậy (Confidence)",
        "lbl_lift": "Độ 'Định Mệnh' (Lift)",
        
        # Nội dung
        "kpi_rules": "Luật tìm thấy",
        "kpi_lift": "Sức mạnh TB",
        "kpi_conf": "Tin cậy Max",
        "chart_scatter": "Bản đồ Phân bố Luật",
        "table_top": "Top Luật Mạnh Nhất",
        "cluster_map": "Bản đồ Khách hàng (PCA)",
        "cluster_stats": "Thống kê RFM",
        "blog_title": "Giải Mã 'Mỏ Vàng' Bán Lẻ",
        "blog_intro": "Dựa trên kết quả phân tích dữ liệu thực tế, chúng tôi đã tìm ra 3 nhóm chiến lược sản phẩm riêng biệt:",
        
        "strat_diamond_name": "💎 Nhóm 1: Luật Kim Cương",
        "strat_diamond_desc": "Các cặp sản phẩm sinh ra là dành cho nhau. Mối quan hệ gần như tuyệt đối.",
        "strat_diamond_act": "Hard Bundle (Đóng gói cứng). Tạo mã SKU mới bán cả bộ, không bán lẻ.",
        
        "strat_gold_name": "🥇 Nhóm 2: Luật Vàng",
        "strat_gold_desc": "Độ tương quan rất cao, thường là các sản phẩm bổ trợ (Ví dụ: Túi thơm + Nến).",
        "strat_gold_act": "Soft Bundle & Recommendation. Hiển thị 'Thường được mua cùng'.",
        
        "strat_silver_name": "🥈 Nhóm 3: Luật Bạc",
        "strat_silver_desc": "Số lượng luật nhiều nhất. Sản phẩm phổ thông hơn.",
        "strat_silver_act": "Discovery & Upsell. Gợi ý mua thêm để được Free-ship."
    },
    "EN": {
        "title": "Customer Intelligence Hub",
        "subtitle": "AI-Driven Segmentation & Association Mining",
        "tabs": ["Overview & Filters", "Customer Segmentation", "Expert Strategy", "Detailed Data"],
        
        "sb_title": "DATA FILTERS",
        "btn_filter": "Apply Filters",
        "lbl_support": "Support",
        "lbl_conf": "Confidence",
        "lbl_lift": "Lift",
        
        "kpi_rules": "Rules Found",
        "kpi_lift": "Avg Strength",
        "kpi_conf": "Max Confidence",
        "chart_scatter": "Association Rules Map",
        "table_top": "Top Strongest Rules",
        "cluster_map": "Customer Map (PCA)",
        "cluster_stats": "RFM Stats",
        "blog_title": "Decoding Retail Insights",
        "blog_intro": "Based on real data analysis, we identified 3 distinct product strategy groups:",
        
        "strat_diamond_name": "💎 Diamond Rules",
        "strat_diamond_desc": "Products born for each other. Almost absolute relationship.",
        "strat_diamond_act": "Hard Bundle. Create new SKU to sell as a set only.",
        
        "strat_gold_name": "🥇 Gold Rules",
        "strat_gold_desc": "Very high correlation, usually complementary products.",
        "strat_gold_act": "Soft Bundle & Recommendation. Show 'Frequently Bought Together'.",
        
        "strat_silver_name": "🥈 Silver Rules",
        "strat_silver_desc": "Most frequent rules. Common products.",
        "strat_silver_act": "Discovery & Upsell. Suggest add-ons for Free-shipping."
    }
}

# --- SIDEBAR LANGUAGE ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3094/3094851.png", width=60)
    lang_option = st.selectbox("Language / Ngôn ngữ:", ["Tiếng Việt", "English"])
    
lang_code = "VN" if lang_option == "Tiếng Việt" else "EN"
T = TRANS[lang_code]

# ==========================================
# 3. CSS (FIX QUAN TRỌNG: HIỆN NÚT SIDEBAR)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Card trắng đổ bóng */
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(255, 255, 255, 1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: #3b82f6;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
    }

    /* Metric & Header */
    .metric-value {
        font-size: 2.2rem; font-weight: 800;
        background: -webkit-linear-gradient(45deg, #0ea5e9, #2563eb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h1, h2, h3 { color: #0f172a !important; }
    
    /* Blog Style */
    .strategy-box {
        padding: 15px; border-radius: 10px; margin-bottom: 15px; border-left: 5px solid;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #3b82f6 100%);
        color: white !important; font-weight: 600; border: none; border-radius: 12px;
        width: 100%;
    }
    
    /* --- FIX LỖI MẤT NÚT SIDEBAR --- */
    /* Ẩn Header mặc định nhưng CHỪA LẠI nút sidebar toggle */
    header[data-testid="stHeader"] {
        background-color: transparent;
    }
    /* Đảm bảo nút này luôn hiện và có màu đậm */
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        color: #1e293b !important;
        background-color: white !important;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        z-index: 1000000;
    }
    
    #MainMenu, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. XỬ LÝ DỮ LIỆU THẬT
# ==========================================
@st.cache_data
def process_real_data():
    try:
        # Load Customers
        path_cust = 'data/processed/customer_clusters_from_rules.csv'
        if not os.path.exists(path_cust): return None, None, f"Lỗi: Không tìm thấy file {path_cust}"
        df_c = pd.read_csv(path_cust)
        
        # Auto-PCA
        if 'PC1' not in df_c.columns:
            features = ['Recency', 'Frequency', 'Monetary']
            X = df_c[features].apply(np.log1p)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=42)
            components = pca.fit_transform(X_scaled)
            df_c['PC1'] = components[:, 0]
            df_c['PC2'] = components[:, 1]

        # Load Rules
        path_rules = 'data/processed/rules_fpgrowth_filtered.csv'
        if not os.path.exists(path_rules):
            path_rules = 'data/processed/rules_apriori_filtered.csv'
        if not os.path.exists(path_rules): return None, None, "Lỗi: Không tìm thấy file Rules."
        
        df_r = pd.read_csv(path_rules)
        # Cleaning
        for col in ['antecedents', 'consequents']:
            if col in df_r.columns:
                df_r[f'{col}_str'] = df_r[col].astype(str).str.replace(r"frozenset\({|}|'|\"", "", regex=True)
                
        return df_c, df_r, None

    except Exception as e:
        return None, None, str(e)

with st.spinner("Đang tải dữ liệu..."):
    df_customers, df_rules, error_msg = process_real_data()

if error_msg:
    st.error(error_msg)
    st.stop()

# ==========================================
# 5. SIDEBAR: BỘ LỌC (CÓ NÚT BẤM)
# ==========================================
with st.sidebar:
    st.markdown(f"### 🎛️ {T['sb_title']}")
    
    # Lấy min/max
    max_lift = float(df_rules['lift'].max())
    max_supp = float(df_rules['support'].max())
    min_supp_real = float(df_rules['support'].min())

    # Form bộ lọc
    with st.form("filter_form"):
        st.caption("Điều chỉnh & bấm nút Áp Dụng:")
        
        min_supp_val = st.slider(T['lbl_support'], 0.0, max_supp, min_supp_real, 0.001, format="%.4f")
        min_lift_val = st.slider(T['lbl_lift'], 0.0, max_lift, 1.0, 0.1)
        min_conf_val = st.slider(T['lbl_conf'], 0.0, 1.0, 0.1, 0.05)
        
        submitted = st.form_submit_button(T['btn_filter'])

    # Logic lọc
    if 'filtered_rules' not in st.session_state:
        st.session_state['filtered_rules'] = df_rules
    
    if submitted:
        st.session_state['filtered_rules'] = df_rules[
            (df_rules['support'] >= min_supp_val) & 
            (df_rules['lift'] >= min_lift_val) & 
            (df_rules['confidence'] >= min_conf_val)
        ]
    
    filtered_rules = st.session_state['filtered_rules']
    st.success(f"Show: **{len(filtered_rules)}** rules")
    st.markdown("---")

# ==========================================
# 6. GIAO DIỆN CHÍNH
# ==========================================
c1, c2 = st.columns([4, 1])
with c1:
    st.markdown(f"<h1 style='margin-bottom:0; color:#0f172a;'>🧬 {T['title']}</h1>", unsafe_allow_html=True)
    st.caption(T['subtitle'])
with c2:
    if lottie_ai: st_lottie(lottie_ai, height=90, key="header_anim")

# --- MENU NGANG (HORIZONTAL) ---
selected = option_menu(
    menu_title=None,
    options=T['tabs'],
    icons=["graph-up-arrow", "people", "star", "table"],
    default_index=0,
    orientation="horizontal", # Menu ngang ở đây
    styles={
        "container": {"background-color": "#ffffff", "border-radius": "12px"},
        "icon": {"color": "#3b82f6", "font-size": "18px"}, 
        "nav-link": {"color": "#475569", "font-size": "15px", "font-weight": "500"},
        "nav-link-selected": {"background-color": "#e0f2fe", "color": "#0284c7", "border": "1px solid #7dd3fc"},
    }
)

# ==========================================
# TAB 1: TỔNG QUAN
# ==========================================
if selected == T['tabs'][0]:
    k1, k2, k3 = st.columns(3)
    def kpi(l, v, d):
        st.markdown(f"<div class='glass-card' style='padding:15px; text-align:center;'><div style='color:#64748b; font-weight:600;'>{l}</div><div class='metric-value'>{v}</div><div style='color:#10b981; font-weight:bold;'>{d}</div></div>", unsafe_allow_html=True)
    
    kpi(T['kpi_rules'], len(df_rules), f"{len(filtered_rules)} filtered")
    kpi(T['kpi_lift'], f"{filtered_rules['lift'].mean():.2f}" if not filtered_rules.empty else "0", "Avg")
    kpi(T['kpi_conf'], f"{filtered_rules['confidence'].max():.0%}" if not filtered_rules.empty else "0%", "Max")

    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown(f'<div class="glass-card"><h5>{T["chart_scatter"]}</h5>', unsafe_allow_html=True)
        if not filtered_rules.empty:
            fig = px.scatter(filtered_rules, x="support", y="confidence", size="lift", color="lift",
                             hover_data=['antecedents_str', 'consequents_str'],
                             color_continuous_scale="Teal", template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Vui lòng mở Sidebar bên trái (>) và giảm bộ lọc.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown(f'<div class="glass-card"><h5>{T["table_top"]}</h5>', unsafe_allow_html=True)
        st.dataframe(
            filtered_rules[['antecedents_str', 'consequents_str', 'lift', 'confidence']].sort_values('lift', ascending=False),
            use_container_width=True, height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: PHÂN CỤM
# ==========================================
elif selected == T['tabs'][1]:
    col_viz, col_stats = st.columns([2, 1])
    with col_viz:
        st.markdown(f'<div class="glass-card"><h5>{T["cluster_map"]}</h5>', unsafe_allow_html=True)
        df_customers['Cluster_Label'] = df_customers['cluster'].astype(str)
        fig_pca = px.scatter(df_customers, x='PC1', y='PC2', color='Cluster_Label',
                             hover_data=['CustomerID', 'Monetary', 'Recency'],
                             color_discrete_sequence=px.colors.qualitative.Bold, opacity=0.8, template="plotly_white", height=500)
        st.plotly_chart(fig_pca, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col_stats:
        st.markdown(f'<div class="glass-card"><h5>{T["cluster_stats"]}</h5>', unsafe_allow_html=True)
        rfm_stats = df_customers.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
        st.dataframe(rfm_stats, use_container_width=True)
        
        st.markdown(f"**{T['cluster_pie']}**")
        fig_pie = px.pie(values=df_customers['cluster'].value_counts(), names=df_customers['cluster'].value_counts().index, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: CHIẾN LƯỢC (BLOG TỪ README)
# ==========================================
elif selected == T['tabs'][2]:
    c_blog, c_side = st.columns([2, 1])
    
    with c_blog:
        st.markdown(f'<div class="glass-card"><h2 style="color:#1e3a8a;">{T["blog_title"]}</h2>', unsafe_allow_html=True)
        st.write(T['blog_intro'])
        
        # Chiến lược 1: Diamond
        st.markdown(f"""
        <div class="strategy-box" style="background-color:#e0f2fe; border-color:#0284c7;">
            <h3 style="color:#0284c7;">{T['strat_diamond_name']}</h3>
            <p><b>🔍 Đặc điểm (Insights):</b> {T['strat_diamond_desc']}</p>
            <p><b>⚡ Hành động (Action):</b> {T['strat_diamond_act']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chiến lược 2: Gold
        st.markdown(f"""
        <div class="strategy-box" style="background-color:#fffbeb; border-color:#d97706;">
            <h3 style="color:#d97706;">{T['strat_gold_name']}</h3>
            <p><b>🔍 Đặc điểm (Insights):</b> {T['strat_gold_desc']}</p>
            <p><b>⚡ Hành động (Action):</b> {T['strat_gold_act']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chiến lược 3: Silver
        st.markdown(f"""
        <div class="strategy-box" style="background-color:#f3f4f6; border-color:#64748b;">
            <h3 style="color:#475569;">{T['strat_silver_name']}</h3>
            <p><b>🔍 Đặc điểm (Insights):</b> {T['strat_silver_desc']}</p>
            <p><b>⚡ Hành động (Action):</b> {T['strat_silver_act']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_side:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Insight Nổi Bật")
        st.info("**Herb Marker Parsley + Rosemary:** Lift = 63.1. Cặp đôi hoàn hảo.")
        st.success("**White Hanging Heart:** Sản phẩm 'Mỏ neo' quốc dân.")
        st.warning("**Cơ hội:** Các luật có Support thấp nhưng Lift cực cao là thị trường ngách.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: DỮ LIỆU
# ==========================================
elif selected == T['tabs'][3]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    t1, t2 = st.tabs(["Customer Data", "Rules Data"])
    with t1: st.dataframe(df_customers, use_container_width=True)
    with t2: st.dataframe(df_rules, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)