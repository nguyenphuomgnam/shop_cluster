import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Shop Cluster Dashboard", layout="wide", page_icon="🛒")

st.title("🛒 Phân khúc Khách hàng & Gợi ý Sản phẩm")
st.markdown("**Mục tiêu:** Phân tích hành vi mua sắm dựa trên Luật kết hợp (Association Rules) và RFM.")

# --- LOAD DỮ LIỆU ---
@st.cache_data
def load_data():
    # Đường dẫn file (Output từ pipeline papermill)
    cluster_path = "data/processed/customer_clusters_from_rules.csv"
    rules_path = "data/processed/rules_apriori_filtered.csv" 
    
    if not os.path.exists(cluster_path) or not os.path.exists(rules_path):
        return None, None
    
    df_c = pd.read_csv(cluster_path)
    df_r = pd.read_csv(rules_path)
    
    # Đảm bảo cột Cluster là string để tô màu cho đẹp
    if 'cluster' in df_c.columns:
        df_c['cluster'] = df_c['cluster'].astype(str)
        
    return df_c, df_r

df_customers, df_rules = load_data()

# --- KIỂM TRA DỮ LIỆU ---
if df_customers is None:
    st.error("⚠️ Chưa tìm thấy file dữ liệu! Hãy chạy lệnh `python run_papermill.py` trước.")
    st.stop()

# --- SIDEBAR: BỘ LỌC ---
st.sidebar.header("🔍 Bộ lọc dữ liệu")
all_clusters = sorted(df_customers['cluster'].unique())
selected_cluster = st.sidebar.selectbox("Chọn Nhóm Khách Hàng (Cluster):", all_clusters)

# Lọc dataset
filtered_df = df_customers[df_customers['cluster'] == selected_cluster]

# --- PHẦN 1: TỔNG QUAN (KPIs) ---
st.subheader(f"📊 Tổng quan Nhóm {selected_cluster}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Số lượng khách", f"{len(filtered_df):,}")
c2.metric("Chi tiêu TB (Monetary)", f"£{filtered_df['Monetary'].mean():,.0f}")
c3.metric("Tần suất mua (Frequency)", f"{filtered_df['Frequency'].mean():.1f} đơn")
c4.metric("Mua gần nhất (Recency)", f"{filtered_df['Recency'].mean():.0f} ngày")

st.divider()

# --- PHẦN 2: TRỰC QUAN HÓA 3D ---
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Mô hình phân cụm 3D (RFM)")
    fig = px.scatter_3d(
        df_customers,
        x='Recency', y='Frequency', z='Monetary',
        color='cluster',
        hover_data=['CustomerID'],
        opacity=0.7,
        title="Không gian phân bố khách hàng"
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Đặc điểm nhóm này")
    st.write(f"Nhóm **{selected_cluster}** có đặc điểm:")
    # Logic mô tả đơn giản
    avg_m = filtered_df['Monetary'].mean()
    if avg_m > df_customers['Monetary'].mean():
        st.success("💰 **Chi tiêu cao (VIP)**")
    else:
        st.warning("💸 **Chi tiêu thấp/Trung bình**")
        
    avg_f = filtered_df['Frequency'].mean()
    if avg_f > df_customers['Frequency'].mean():
        st.info("🔄 **Mua sắm thường xuyên**")
    else:
        st.info("zzz **Ít mua sắm**")

# --- PHẦN 3: LUẬT KẾT HỢP & GỢI Ý ---
st.divider()
st.subheader("💡 Top Luật mua sắm (Dùng để Cross-sell)")

# Hiển thị Top luật có Lift cao nhất
top_rules = df_rules.sort_values('lift', ascending=False).head(10)
st.dataframe(
    top_rules[['antecedents_str', 'consequents_str', 'confidence', 'lift', 'support']],
    column_config={
        "antecedents_str": "Khách mua cái này...",
        "consequents_str": "...Sẽ mua thêm cái này",
        "confidence": "Độ tin cậy",
        "lift": "Độ nâng (Lift)",
        "support": "Độ phổ biến"
    },
    use_container_width=True,
    hide_index=True
)

# --- PHẦN 4: DỮ LIỆU CHI TIẾT ---
with st.expander("Xem danh sách khách hàng chi tiết"):
    st.dataframe(filtered_df)