"""
Multi-Layer Intrusion Detection System - Streamlit Web App
===========================================================

A web interface for detecting:
1. SYN Scan attacks (Network Layer)
2. SQL Injection attacks (Application Layer)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import preprocessing functions
from data.preprocess import preprocess_pipeline
from data.preprocess_sqli import preprocess_http_pipeline
from features.build_features_sqli import extract_features_batch as extract_sqli_features

# Import core prediction functions from predict.py (REUSE!)
from predict import predict_syn_scan_from_df, predict_sqli_from_df


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multi-Layer IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .attack-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
    .normal-alert {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================

def check_models_exist():
    """Check if trained models exist."""
    models_status = {}
    
    syn_path = Path(__file__).parent / "models" / "syn_scan_detector.joblib"
    sqli_path = Path(__file__).parent / "models" / "sqli_detector.joblib"
    
    models_status['syn_scan'] = syn_path.exists()
    models_status['sqli'] = sqli_path.exists()
    
    return models_status


# ============================================================
# DETECTION FUNCTIONS (Using predict.py core functions)
# ============================================================

def detect_syn_scan(df_raw):
    """Detect SYN scan attacks using predict.py core function."""
    try:
        # Save to temp file for preprocessing (preprocessing requires file path)
        temp_path = Path(__file__).parent / "temp_upload.csv"
        df_raw.to_csv(temp_path, index=False)
        
        # Preprocess
        aggregated_df = preprocess_pipeline(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        if len(aggregated_df) == 0:
            return None, "No data after preprocessing"
        
        # Use core prediction function from predict.py
        result_df = predict_syn_scan_from_df(aggregated_df)
        
        # Add display status (with emoji) for UI
        result_df['status'] = result_df['prediction'].map({0: '✅ Normal', 1: '⚠️ SYN Scan'})
        result_df['status_export'] = result_df['prediction_label']  # Clean text for export
        
        return result_df, None
        
    except Exception as e:
        return None, str(e)


def detect_sqli(df_raw):
    """Detect SQL injection attacks using predict.py core function."""
    try:
        # Save to temp file for preprocessing
        temp_path = Path(__file__).parent / "temp_upload.csv"
        df_raw.to_csv(temp_path, index=False)
        
        # Preprocess
        http_df = preprocess_http_pipeline(str(temp_path))
        
        # Clean up
        temp_path.unlink()
        
        if len(http_df) == 0:
            return None, "No HTTP requests found"
        
        # Extract features
        features_df = extract_sqli_features(http_df)
        
        # Use core prediction function from predict.py
        result_df = predict_sqli_from_df(features_df)
        
        # Add display status (with emoji) for UI
        result_df['status'] = result_df['prediction'].map({0: '✅ Normal', 1: '⚠️ SQLi Attack'})
        result_df['status_export'] = result_df['prediction_label']  # Clean text for export
        
        return result_df, None
        
    except Exception as e:
        return None, str(e)


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_pie_chart(attack_count, normal_count, title):
    """Create a pie chart for attack distribution."""
    fig = go.Figure(data=[go.Pie(
        labels=['Normal', 'Attack'],
        values=[normal_count, attack_count],
        hole=0.4,
        marker_colors=['#4CAF50', '#F44336']
    )])
    fig.update_layout(
        title=title,
        showlegend=True,
        height=300
    )
    return fig


def create_timeline_chart(df, time_col, prediction_col, title):
    """Create a timeline chart showing attacks over time."""
    if time_col not in df.columns:
        return None
    
    df_plot = df.copy()
    df_plot['time'] = pd.to_datetime(df_plot[time_col], unit='s')
    
    fig = px.scatter(
        df_plot,
        x='time',
        y=prediction_col,
        color='status',
        color_discrete_map={'✅ Normal': '#4CAF50', '⚠️ SYN Scan': '#F44336', '⚠️ SQLi Attack': '#F44336'},
        title=title
    )
    fig.update_layout(height=300)
    return fig


def create_feature_importance_chart(model, feature_names):
    """Create feature importance chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    fig = go.Figure(go.Bar(
        x=[importances[i] for i in indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker_color='#1E88E5'
    ))
    fig.update_layout(
        title="Top 10 Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<p class="main-header">🛡️ Multi-Layer Intrusion Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Detect SYN Scan and SQL Injection attacks from network traffic</p>', unsafe_allow_html=True)
    
    # Check model availability
    models_status = check_models_exist()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Model status
    st.sidebar.subheader("Model Status")
    if models_status['syn_scan']:
        st.sidebar.success("✅ SYN Scan Model Loaded")
    else:
        st.sidebar.error("❌ SYN Scan Model Not Found")
    
    if models_status['sqli']:
        st.sidebar.success("✅ SQLi Model Loaded")
    else:
        st.sidebar.error("❌ SQLi Model Not Found")
    
    # Detection options
    st.sidebar.subheader("Detection Options")
    detect_syn = st.sidebar.checkbox("Detect SYN Scans", value=True, disabled=not models_status['syn_scan'])
    detect_sql = st.sidebar.checkbox("Detect SQL Injection", value=True, disabled=not models_status['sqli'])
    
    # Main content
    st.markdown("---")
    
    # File upload
    st.subheader("📁 Upload Network Capture")
    st.markdown("""
    Upload a CSV file exported from Wireshark with the following columns:
    - `frame.time_epoch` - Timestamp
    - `ip.src`, `ip.dst` - Source/Destination IP
    - `tcp.flags.syn`, `tcp.flags.ack` - TCP flags (for SYN scan)
    - `tcp.dstport` - Destination port
    - `http.request.uri` - HTTP URI (for SQL injection)
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        try:
            df_raw = pd.read_csv(uploaded_file, on_bad_lines='skip')
            st.success(f"✅ Loaded {len(df_raw)} packets")
            
            # Show preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df_raw.head(100), use_container_width=True)
            
            # Run detection
            if st.button("🔍 Run Detection", type="primary", use_container_width=True):
                
                col1, col2 = st.columns(2)
                
                # SYN Scan Detection
                if detect_syn and models_status['syn_scan']:
                    with col1:
                        st.subheader("🌐 SYN Scan Detection")
                        with st.spinner("Analyzing network traffic..."):
                            syn_results, syn_error = detect_syn_scan(df_raw)
                        
                        if syn_error:
                            st.error(f"Error: {syn_error}")
                        elif syn_results is not None:
                            attack_count = syn_results['prediction'].sum()
                            normal_count = len(syn_results) - attack_count
                            
                            # Metrics
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Total Windows", len(syn_results))
                            m2.metric("Attacks", int(attack_count), delta=None if attack_count == 0 else "⚠️")
                            m3.metric("Normal", int(normal_count))
                            
                            # Alert
                            if attack_count > 0:
                                st.markdown(f"""
                                <div class="attack-alert">
                                    <strong>⚠️ SYN Scan Attacks Detected!</strong><br>
                                    {int(attack_count)} time window(s) show port scanning activity.
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="normal-alert">
                                    <strong>✅ No SYN Scan Detected</strong><br>
                                    Network traffic appears normal.
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Pie chart
                            fig = create_pie_chart(attack_count, normal_count, "SYN Scan Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Results table
                            with st.expander("📋 Detailed Results"):
                                display_cols = ['ip.src', 'ip.dst', 'unique_dst_ports', 'syn_count', 'attack_probability', 'status']
                                display_cols = [c for c in display_cols if c in syn_results.columns]
                                st.dataframe(syn_results[display_cols].head(50), use_container_width=True)
                            
                            # Download results (without emoji status)
                            export_df = syn_results.copy()
                            export_df['status'] = export_df['status_export']
                            export_df = export_df.drop(columns=['status_export'])
                            export_df = export_df.drop(columns=['prediction_label'])
                            csv = export_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download SYN Scan Results",
                                csv,
                                "syn_scan_results.csv",
                                "text/csv"
                            )
                
                # SQL Injection Detection
                if detect_sql and models_status['sqli']:
                    with col2:
                        st.subheader("💉 SQL Injection Detection")
                        with st.spinner("Analyzing HTTP requests..."):
                            sqli_results, sqli_error = detect_sqli(df_raw)
                        
                        if sqli_error:
                            st.error(f"Error: {sqli_error}")
                        elif sqli_results is not None:
                            attack_count = sqli_results['prediction'].sum()
                            normal_count = len(sqli_results) - attack_count
                            
                            # Metrics
                            m1, m2, m3 = st.columns(3)
                            m1.metric("HTTP Requests", len(sqli_results))
                            m2.metric("Attacks", int(attack_count), delta=None if attack_count == 0 else "⚠️")
                            m3.metric("Normal", int(normal_count))
                            
                            # Alert
                            if attack_count > 0:
                                st.markdown(f"""
                                <div class="attack-alert">
                                    <strong>⚠️ SQL Injection Attacks Detected!</strong><br>
                                    {int(attack_count)} malicious HTTP request(s) found.
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="normal-alert">
                                    <strong>✅ No SQL Injection Detected</strong><br>
                                    HTTP traffic appears safe.
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Pie chart
                            fig = create_pie_chart(attack_count, normal_count, "SQLi Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Results table
                            with st.expander("📋 Detailed Results"):
                                display_cols = ['ip.src', 'ip.dst', 'decoded_uri', 'has_sql_keywords', 'attack_probability', 'status']
                                display_cols = [c for c in display_cols if c in sqli_results.columns]
                                st.dataframe(sqli_results[display_cols].head(50), use_container_width=True)
                            
                            # Download results (without emoji status)
                            export_df = sqli_results.copy()
                            export_df['status'] = export_df['status_export']
                            export_df = export_df.drop(columns=['status_export'])
                            export_df = export_df.drop(columns=['prediction_label'])
                            csv = export_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download SQLi Results",
                                csv,
                                "sqli_results.csv",
                                "text/csv"
                            )
        
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        Multi-Layer IDS | SYN Scan Detection (Network Layer) + SQL Injection Detection (Application Layer)<br>
        Built with Streamlit & Scikit-learn
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
