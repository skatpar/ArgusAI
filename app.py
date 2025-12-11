"""
ArgusAI - Fraud Detection & Monitoring Platform
A comprehensive dashboard for fraud detection, feature monitoring, model monitoring, and case management
"""

import streamlit as st
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="ArgusAI - Fraud Detection Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional theme with #744ada, black, grey
st.markdown("""
    <style>
    /* Color Palette */
    :root {
        --primary-purple: #744ada;
        --dark-purple: #5a38ad;
        --light-purple: #9b7fe8;
        --black: #000000;
        --dark-grey: #2b2b2b;
        --medium-grey: #666666;
        --light-grey: #cccccc;
        --bg-grey: #f5f5f5;
    }

    /* Main Headers */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--black);
        text-align: left;
        padding: 0.5rem 0;
        border-bottom: 3px solid var(--primary-purple);
        margin-bottom: 2rem;
        letter-spacing: -0.5px;
    }

    .sub-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--dark-grey);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-purple);
        padding-left: 1rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-grey) 0%, #ffffff 100%);
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid var(--light-grey);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: var(--bg-grey);
        padding: 0.5rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 500;
        color: var(--dark-grey);
        background-color: transparent;
        border-radius: 6px;
        border: none;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(116, 74, 218, 0.1);
        color: var(--primary-purple);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-purple) !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--primary-purple);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--dark-purple);
        box-shadow: 0 4px 8px rgba(116, 74, 218, 0.3);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--dark-grey);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--primary-purple);
        font-weight: 600;
    }

    /* Dataframes */
    .dataframe {
        border: 1px solid var(--light-grey) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--bg-grey);
        border-radius: 6px;
        font-weight: 500;
        color: var(--dark-grey);
    }

    /* Info boxes */
    .stAlert {
        background-color: rgba(116, 74, 218, 0.1);
        border-left: 4px solid var(--primary-purple);
        border-radius: 6px;
    }

    /* Remove default streamlit branding colors */
    .stApp {
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; border-bottom: 2px solid #744ada;'>
            <h1 style='color: white; font-size: 1.8rem; font-weight: 600; margin: 0; letter-spacing: 1px;'>ARGUS<span style='color: #744ada;'>AI</span></h1>
            <p style='color: #cccccc; font-size: 0.85rem; margin: 0.5rem 0 0 0; font-weight: 400;'>Fraud Detection Platform</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Feature Monitoring", "Model Monitoring", "Rule Editor", "Case Management", "Model Deployment"],
        icons=["graph-up-arrow", "activity", "shield-check", "folder-open", "cpu"],
        menu_icon=None,
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "#2b2b2b"},
            "icon": {"color": "#744ada", "font-size": "1.1rem"},
            "nav-link": {
                "color": "#cccccc",
                "font-size": "0.95rem",
                "font-weight": "400",
                "text-align": "left",
                "margin": "0.2rem 0",
                "padding": "0.8rem 1rem",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background-color": "#744ada",
                "color": "white",
                "font-weight": "500",
            },
        }
    )

    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div style='background-color: #1a1a1a; padding: 1.2rem; border-radius: 8px; border: 1px solid #444;'>
            <p style='color: #744ada; font-weight: 600; font-size: 0.9rem; margin: 0 0 0.8rem 0;'>PLATFORM CAPABILITIES</p>
            <ul style='color: #cccccc; font-size: 0.85rem; line-height: 1.8; margin: 0; padding-left: 1.2rem;'>
                <li>Feature drift & data quality monitoring</li>
                <li>Model performance & fraud KPI tracking</li>
                <li>Rule-based fraud detection</li>
                <li>Case investigation & management</li>
                <li>Model deployment & inference</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

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
