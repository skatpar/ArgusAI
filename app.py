"""
ArgusAI - Fraud Detection & Monitoring Platform
A comprehensive dashboard for fraud detection, feature monitoring, model monitoring, and case management
"""

import streamlit as st
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="ArgusAI - Fraud Detection Platform",
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
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🛡️ ArgusAI")
    st.markdown("### Fraud Detection Platform")
    st.markdown("---")

    selected = option_menu(
        menu_title="Navigation",
        options=["Feature Monitoring", "Model Monitoring", "Rule Editor", "Case Management", "Model Deployment"],
        icons=["bar-chart-line", "speedometer2", "list-check", "search", "cpu"],
        menu_icon="cast",
        default_index=0,
    )

    st.markdown("---")
    st.markdown("### About")
    st.info("""
        ArgusAI provides a complete platform for:
        - Feature drift & data quality monitoring
        - Model performance & fraud KPI tracking
        - Rule-based fraud detection
        - Case investigation & management
        - Model deployment & inference
    """)

# Main content
if selected == "Feature Monitoring":
    from src.pages import feature_monitoring
    feature_monitoring.show()

elif selected == "Model Monitoring":
    from src.pages import model_monitoring_detailed
    model_monitoring_detailed.show()

elif selected == "Rule Editor":
    from src.pages import rule_editor
    rule_editor.show()

elif selected == "Case Management":
    from src.pages import case_management
    case_management.show()

elif selected == "Model Deployment":
    from src.pages import model_deployment
    model_deployment.show()
