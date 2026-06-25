import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Signal EDA & Peak Clustering",
    page_icon="📈",
    layout="wide"
)

# =========================
# Sidebar: Logo and Developer
# =========================
st.sidebar.image(
    "https://brand.umpsa.edu.my/images/logo-umpsa-full-color2.png", 
    use_container_width=True
)
st.sidebar.image(
    "https://www.majalahsains.com/wp-content/uploads/2012/05/Logo-Agensi-Nuklear-Malaysia.png",
    use_container_width=True
)    

st.sidebar.markdown("## EDA & Peak Clustering Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("### Developers:")
st.sidebar.write("***Assoc. Prof. Dr. Ku Muhammad Naim Ku Khalif***")
st.sidebar.write("Centre for Mathematical Sciences\nUniversiti Malaysia Pahang Al-Sultan Abdullah\nEmail: kunaim@umpsa.edu.my")

st.sidebar.write("***Dr. Nurul A'in binti Ahmad Latif***")
st.sidebar.write("Bahagian Teknologi Industri (BTI)\nAgensi Nuklear Malaysia\nEmail: nurul_ain@nm.gov.my")
st.sidebar.markdown("---")

# =========================
# Main title
# =========================
st.title("Signal Data Analysis & Peak Clustering")
st.caption("Upload multiple text, CSV, or Excel data files to explore signals, find local maxima, and cluster peak values.")

# =========================
# File Upload
# =========================
st.sidebar.header("Upload Dataset(s)")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, CSV, or Excel files",
    type=["txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

@st.cache_data
def load_data(file):
    # Handle the specific .txt format provided in the examples (space delimited, % commented header)
    if file.name.endswith(".txt"):
        df = pd.read_csv(file, sep='\s+', comment='%', names=['X', 'Y', 'Z'])
        # Drop columns that might be completely empty
        df = df.dropna(axis=1, how='all')
        return df
    elif file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# =========================
# Main App Body
# =========================
if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} file(s).")
    
    # Load all data files
    all_data = {}
    for file in uploaded_files:
        all_data[file.name] = load_data(file)
    
    # Create tabs to separate standard EDA from the specific Peak processing
    tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "⛰️ Peak Detection & Clustering"])

    with tab1:
        st.subheader("Combined Signals Visualization")
        
        # Get all numeric columns across all files
        all_numeric_cols = set()
        for df in all_data.values():
            all_numeric_cols.update(df.select_dtypes(include=np.number).columns.tolist())
        all_numeric_cols = sorted(list(all_numeric_cols))
        
        if len(all_numeric_cols) >= 1:
            x_col = st.selectbox("Select X-axis", all_numeric_cols, index=0, key="eda_x")
            y_col = st.selectbox("Select Y-axis", all_numeric_cols, index=1 if len(all_numeric_cols) > 1 else 0, key="eda_y")
            
            # Create combined plot with all signals
            fig = go.Figure()
            
            color_palette = px.colors.qualitative.Plotly
            for idx, (file_name, df) in enumerate(all_data.items()):
                if x_col in df.columns and y_col in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[x_col],
                        y=df[y_col],
                        mode='lines',
                        name=file_name,
                        line=dict(color=color_palette[idx % len(color_palette)], width=2)
                    ))
            
            fig.update_layout(
                title=f"Combined Signals: {y_col} vs {x_col}",
                xaxis_title=x_col,
                yaxis_title=y_col,
                hovermode="x unified",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found in uploaded files.")
        
        # Individual file statistics
        st.subheader("Individual File Statistics")
        for file_name, df in all_data.items():
            st.markdown(f"**{file_name}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", int(df.isnull().sum().sum()))
            
            with st.expander(f"View {file_name} data"):
                st.dataframe(df.head(50), use_container_width=True)

    with tab2:
        st.subheader("Peak Detection & Clustering - All Files Combined")
        
        if len(all_numeric_cols) >= 1:
            row1_col1, row1_col2 = st.columns(2)
            
            with row1_col1:
                st.markdown("**1. Configure Peak Detection**")
                x_peak = st.selectbox("Select X-axis (Time/Index)", all_numeric_cols, index=0, key="peak_x")
                y_peak = st.selectbox("Select Y-axis (Signal Amplitude)", all_numeric_cols, index=1 if len(all_numeric_cols) > 1 else 0, key="peak_y")
                
                # Parameters for scipy.signal.find_peaks
                prominence = st.number_input("Prominence (Minimum peak height relative to baseline)", value=0.01, step=0.01)
                distance = st.number_input("Minimum horizontal distance between peaks (rows)", value=1, step=1, min_value=1)
                
            with row1_col2:
                st.markdown("**2. Configure Clustering**")
                n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
                cluster_feature = st.radio("Cluster peaks based on:", ["Y Amplitude only", "X and Y coordinates"])

            # Combine all peaks from all files
            all_peaks_x = []
            all_peaks_y = []
            file_labels = []
            
            fig_combined = go.Figure()
            color_palette = px.colors.qualitative.Plotly
            
            for file_idx, (file_name, df) in enumerate(all_data.items()):
                if x_peak not in df.columns or y_peak not in df.columns:
                    st.warning(f"Skipping {file_name}: missing required columns")
                    continue
                
                y_data = df[y_peak].values
                x_data = df[x_peak].values
                
                # Add signal line to figure
                fig_combined.add_trace(go.Scatter(
                    x=x_data, y=y_data,
                    mode='lines',
                    name=f'{file_name} (Signal)',
                    line=dict(color=color_palette[file_idx % len(color_palette)], width=1.5),
                    opacity=0.7
                ))
                
                # Find peaks
                peaks_idx, _ = find_peaks(y_data, prominence=prominence, distance=distance)
                
                if len(peaks_idx) > 0:
                    peak_x = x_data[peaks_idx]
                    peak_y = y_data[peaks_idx]
                    
                    all_peaks_x.extend(peak_x)
                    all_peaks_y.extend(peak_y)
                    file_labels.extend([file_name] * len(peaks_idx))
            
            if len(all_peaks_x) == 0:
                st.warning("No peaks found with the current parameters. Try lowering the prominence.")
            else:
                all_peaks_x = np.array(all_peaks_x)
                all_peaks_y = np.array(all_peaks_y)
                
                # Clustering
                if cluster_feature == "Y Amplitude only":
                    X_cluster = all_peaks_y.reshape(-1, 1)
                else:
                    X_cluster = np.column_stack((all_peaks_x, all_peaks_y))
                
                # KMeans requires at least as many samples as clusters
                if len(all_peaks_x) < n_clusters:
                    st.warning(f"Found only {len(all_peaks_x)} peaks total. Lowering clusters to {len(all_peaks_x)}.")
                    n_clusters = len(all_peaks_x)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(X_cluster)
                
                # Plot Peaks colored by Cluster
                color_scale = px.colors.qualitative.Plotly
                for cluster_id in range(n_clusters):
                    mask = (labels == cluster_id)
                    fig_combined.add_trace(go.Scatter(
                        x=all_peaks_x[mask],
                        y=all_peaks_y[mask],
                        mode='markers',
                        marker=dict(size=10, symbol='star', line=dict(width=1), color=color_scale[cluster_id % len(color_scale)]),
                        name=f'Cluster {cluster_id}',
                        text=[file_labels[i] for i in np.where(mask)[0]],
                        hovertemplate='<b>%{text}</b><br>X: %{x}<br>Y: %{y}<extra></extra>'
                    ))
                
                fig_combined.update_layout(
                    title=f"Detected {len(all_peaks_x)} Total Peaks grouped into {n_clusters} Clusters",
                    xaxis_title=x_peak,
                    yaxis_title=y_peak,
                    hovermode="closest",
                    height=600
                )
                
                st.plotly_chart(fig_combined, use_container_width=True)
                
                # Show Peak Data
                st.markdown("### Peak Data Summary")
                peak_df = pd.DataFrame({
                    'File': file_labels,
                    x_peak: all_peaks_x,
                    y_peak: all_peaks_y,
                    "Cluster Label": labels
                })
                st.dataframe(peak_df, use_container_width=True)
                
                # Download Peak Data
                csv = peak_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Combined Peak Data as CSV",
                    data=csv,
                    file_name="combined_clustered_peaks.csv",
                    mime="text/csv",
                )
        else:
            st.warning("No numeric columns found in uploaded files.")
else:
    st.info("👈 Please upload TXT, CSV, or Excel dataset(s) from the sidebar to begin.")
