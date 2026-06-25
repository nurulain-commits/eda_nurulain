import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import io

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
st.caption("Upload text, CSV, or Excel data to explore signals, find local maxima, and cluster peak values.")

# =========================
# File Upload
# =========================
st.sidebar.header("Upload Dataset")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, CSV, or Excel files (multiple allowed)",
    type=["txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

def load_data(file):
    """Load data from uploaded file - handles TXT, CSV, and Excel formats"""
    try:
        if file.name.endswith(".txt"):
            # Read file content as string
            file_content = file.read().decode('utf-8')
            # Use StringIO to create a file-like object
            # Use sep='\s+' instead of deprecated delim_whitespace
            df = pd.read_csv(
                io.StringIO(file_content), 
                sep='\s+',
                comment='%', 
                names=['X', 'Y', 'Z']
            )
            # Drop columns that might be completely empty
            df = df.dropna(axis=1, how='all')
            return df
        elif file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:  # Excel files
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# =========================
# Main App Body
# =========================
if uploaded_files:
    # Create file selection
    file_names = [f.name for f in uploaded_files]
    selected_file_idx = st.sidebar.selectbox(
        "Select file to analyze:",
        range(len(uploaded_files)),
        format_func=lambda x: file_names[x]
    )
    
    # Load selected file
    selected_file = uploaded_files[selected_file_idx]
    df = load_data(selected_file)
    
    if df is not None:
        st.success(f"✅ Dataset loaded: **{selected_file.name}** ({df.shape[0]} rows, {df.shape[1]} columns)")

        # Create tabs to separate standard EDA from the specific Peak processing
        tab1, tab2 = st.tabs(["📊 Exploratory Data Analysis", "⛰️ Peak Detection & Clustering"])

        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(100), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing Values", int(df.isnull().sum().sum()))

            st.subheader("Descriptive Statistics")
            st.dataframe(df.describe().T, use_container_width=True)

            st.subheader("Data Visualization")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("Select X-axis", numeric_cols, index=0, key="eda_x")
                y_col = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="eda_y")
                
                fig_line = px.line(df, x=x_col, y=y_col, title=f"Line Plot: {y_col} vs {x_col}")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("Need at least two numeric columns to plot X vs Y.")

        with tab2:
            st.subheader("Peak Detection & Clustering")
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) >= 2:
                row1_col1, row1_col2 = st.columns(2)
                
                with row1_col1:
                    st.markdown("**1. Configure Peak Detection**")
                    x_peak = st.selectbox("Select X-axis (Time/Index)", numeric_cols, index=0, key="peak_x")
                    y_peak = st.selectbox("Select Y-axis (Signal Amplitude)", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="peak_y")
                    
                    # Parameters for scipy.signal.find_peaks
                    prominence = st.number_input("Prominence (Minimum peak height relative to baseline)", value=0.01, step=0.01)
                    distance = st.number_input("Minimum horizontal distance between peaks (rows)", value=1, step=1, min_value=1)
                    
                with row1_col2:
                    st.markdown("**2. Configure Clustering**")
                    n_clusters = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=3)
                    cluster_feature = st.radio("Cluster peaks based on:", ["Y Amplitude only", "X and Y coordinates"])

                # Processing Peaks
                y_data = df[y_peak].values
                x_data = df[x_peak].values
                
                peaks_idx, _ = find_peaks(y_data, prominence=prominence, distance=distance)
                
                if len(peaks_idx) == 0:
                    st.warning("No peaks found with the current parameters. Try lowering the prominence.")
                else:
                    peak_x = x_data[peaks_idx]
                    peak_y = y_data[peaks_idx]
                    
                    # Clustering
                    if cluster_feature == "Y Amplitude only":
                        X_cluster = peak_y.reshape(-1, 1)
                    else:
                        X_cluster = np.column_stack((peak_x, peak_y))
                        
                    # KMeans requires at least as many samples as clusters
                    if len(peaks_idx) < n_clusters:
                        st.warning(f"Found only {len(peaks_idx)} peaks. Lowering clusters to {len(peaks_idx)}.")
                        n_clusters = len(peaks_idx)

                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                    labels = kmeans.fit_predict(X_cluster)
                    
                    # Plotting results
                    st.markdown("### Peak Clustering Visualization")
                    
                    fig = go.Figure()
                    
                    # Base Signal
                    fig.add_trace(go.Scatter(
                        x=x_data, y=y_data, 
                        mode='lines', 
                        name='Original Signal',
                        line=dict(color='gray', width=1.5),
                        opacity=0.7
                    ))
                    
                    # Plot Peaks colored by Cluster
                    color_scale = px.colors.qualitative.Plotly
                    for cluster_id in range(n_clusters):
                        mask = (labels == cluster_id)
                        fig.add_trace(go.Scatter(
                            x=peak_x[mask], 
                            y=peak_y[mask],
                            mode='markers',
                            marker=dict(size=12, symbol='x', line=dict(width=2), color=color_scale[cluster_id % len(color_scale)]),
                            name=f'Cluster {cluster_id}'
                        ))
                    
                    fig.update_layout(
                        title=f"Detected {len(peaks_idx)} Peaks grouped into {n_clusters} Clusters",
                        xaxis_title=x_peak,
                        yaxis_title=y_peak,
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show Peak Data
                    st.markdown("### Peak Data Summary")
                    peak_df = pd.DataFrame({
                        x_peak: peak_x,
                        y_peak: peak_y,
                        "Cluster Label": labels
                    })
                    st.dataframe(peak_df, use_container_width=True)
                    
                    # Download Peak Data
                    csv = peak_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Peak Data as CSV",
                        data=csv,
                        file_name=f"clustered_peaks_{selected_file.name.split('.')[0]}.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("Need at least two numeric columns to perform peak detection and clustering.")
else:
    st.info("👈 Please upload one or more TXT, CSV, or Excel datasets from the sidebar to begin.")
