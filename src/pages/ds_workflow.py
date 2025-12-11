"""
Data Science Workflow Page
Complete end-to-end workflow from data extraction to model evaluation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import sys
sys.path.append('/home/user/ArgusAI')
from src.utils.data_generator import generate_fraud_data
from src.utils.feature_engineering import create_features, get_feature_importance
import warnings
warnings.filterwarnings('ignore')


def show():
    st.markdown('<p class="main-header">Data Science Workflow</p>', unsafe_allow_html=True)
    st.markdown("Complete end-to-end workflow for fraud detection modeling")

    # Create tabs for different workflow stages
    tabs = st.tabs([
        "📥 Data Extraction",
        "🧹 Data Cleaning",
        "EDA",
        "⚙Feature Engineering",
        "Feature Selection",
        "👥 Customer Profiling",
        "🔀 Train/Test Split",
        "Model Building",
        "Model Evaluation",
        "📋 Final Metrics"
    ])

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'df_featured' not in st.session_state:
        st.session_state.df_featured = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

    # Tab 1: Data Extraction
    with tabs[0]:
        show_data_extraction()

    # Tab 2: Data Cleaning
    with tabs[1]:
        show_data_cleaning()

    # Tab 3: EDA
    with tabs[2]:
        show_eda()

    # Tab 4: Feature Engineering
    with tabs[3]:
        show_feature_engineering()

    # Tab 5: Feature Selection
    with tabs[4]:
        show_feature_selection()

    # Tab 6: Customer Profiling
    with tabs[5]:
        show_customer_profiling()

    # Tab 7: Train/Test Split
    with tabs[6]:
        show_train_test_split()

    # Tab 8: Model Building
    with tabs[7]:
        show_model_building()

    # Tab 9: Model Evaluation
    with tabs[8]:
        show_model_evaluation()

    # Tab 10: Final Metrics
    with tabs[9]:
        show_final_metrics()


def show_data_extraction():
    st.markdown("### 📥 Data Extraction")
    st.markdown("Load and extract data from various sources")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Data Source Options")
        source = st.radio(
            "Select data source:",
            ["Generate Sample Data", "Upload CSV", "Database Connection (Simulated)"]
        )

    with col2:
        st.markdown("#### Sample Size")
        n_samples = st.number_input("Number of records:", min_value=100, max_value=100000, value=10000, step=1000)
        fraud_rate = st.slider("Fraud rate (%):", min_value=1, max_value=20, value=5)

    if st.button("🔄 Extract Data", type="primary"):
        with st.spinner("Extracting data..."):
            if source == "Generate Sample Data":
                st.session_state.df = generate_fraud_data(n_samples=n_samples, fraud_rate=fraud_rate/100)
                st.success(f"Successfully generated {len(st.session_state.df)} records!")
            elif source == "Upload CSV":
                st.info("CSV upload will be available in the file uploader below")
            else:
                st.session_state.df = generate_fraud_data(n_samples=n_samples, fraud_rate=fraud_rate/100)
                st.success(f"Successfully extracted {len(st.session_state.df)} records from database!")

    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(st.session_state.df)} records!")

    if st.session_state.df is not None:
        st.markdown("---")
        st.markdown("#### 📋 Data Preview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(st.session_state.df):,}")
        with col2:
            st.metric("Total Features", len(st.session_state.df.columns))
        with col3:
            fraud_count = st.session_state.df['is_fraud'].sum()
            st.metric("Fraud Cases", f"{fraud_count:,}")
        with col4:
            fraud_pct = (fraud_count / len(st.session_state.df)) * 100
            st.metric("Fraud Rate", f"{fraud_pct:.2f}%")

        st.dataframe(st.session_state.df.head(20), use_container_width=True)

        st.markdown("#### Data Info")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types:**")
            st.dataframe(pd.DataFrame({
                'Column': st.session_state.df.dtypes.index,
                'Type': st.session_state.df.dtypes.values
            }), use_container_width=True)

        with col2:
            st.markdown("**Missing Values:**")
            missing = st.session_state.df.isnull().sum()
            st.dataframe(pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Percentage': (missing.values / len(st.session_state.df) * 100).round(2)
            }), use_container_width=True)


def show_data_cleaning():
    st.markdown("### 🧹 Data Cleaning")
    st.markdown("Clean and preprocess the extracted data")

    if st.session_state.df is None:
        st.warning("Please extract data first from the 'Data Extraction' tab")
        return

    df = st.session_state.df.copy()

    st.markdown("#### Cleaning Operations")

    col1, col2 = st.columns(2)
    with col1:
        remove_duplicates = st.checkbox("Remove duplicate records", value=True)
        handle_missing = st.checkbox("Handle missing values", value=True)
        remove_outliers = st.checkbox("Remove outliers", value=False)

    with col2:
        if handle_missing:
            missing_strategy = st.selectbox(
                "Missing value strategy:",
                ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"]
            )

        if remove_outliers:
            outlier_std = st.slider("Outlier threshold (std dev):", 2.0, 5.0, 3.0, 0.5)

    if st.button("🧹 Clean Data", type="primary"):
        with st.spinner("Cleaning data..."):
            original_shape = df.shape

            # Remove duplicates
            if remove_duplicates:
                df = df.drop_duplicates()

            # Handle missing values
            if handle_missing:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if missing_strategy == "Drop rows":
                    df = df.dropna()
                elif missing_strategy == "Fill with mean":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif missing_strategy == "Fill with median":
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif missing_strategy == "Fill with mode":
                    for col in df.columns:
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 0)

            # Remove outliers
            if remove_outliers:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'is_fraud']
                for col in numeric_cols:
                    mean = df[col].mean()
                    std = df[col].std()
                    df = df[(df[col] >= mean - outlier_std * std) & (df[col] <= mean + outlier_std * std)]

            st.session_state.df_cleaned = df

            st.success(f"""
                Data cleaning completed!
                - Original shape: {original_shape}
                - Cleaned shape: {df.shape}
                - Rows removed: {original_shape[0] - df.shape[0]}
            """)

    if st.session_state.df_cleaned is not None:
        st.markdown("---")
        st.markdown("#### Cleaned Data Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records After Cleaning", f"{len(st.session_state.df_cleaned):,}")
        with col2:
            st.metric("Features", len(st.session_state.df_cleaned.columns))
        with col3:
            missing_total = st.session_state.df_cleaned.isnull().sum().sum()
            st.metric("Missing Values", missing_total)
        with col4:
            duplicates = st.session_state.df_cleaned.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)

        st.dataframe(st.session_state.df_cleaned.head(20), use_container_width=True)


def show_eda():
    st.markdown("### Exploratory Data Analysis")
    st.markdown("Analyze and visualize data patterns")

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df

    if df is None:
        st.warning("Please extract data first")
        return

    # Summary Statistics
    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # Visualizations
    st.markdown("---")
    st.markdown("#### Visualizations")

    viz_type = st.selectbox(
        "Select visualization:",
        ["Fraud Distribution", "Feature Distributions", "Correlation Heatmap",
         "Fraud by Category", "Amount Distribution", "Time Series Analysis"]
    )

    if viz_type == "Fraud Distribution":
        fig = px.pie(df, names='is_fraud', title='Fraud vs Non-Fraud Distribution',
                     labels={'is_fraud': 'Fraud Status'},
                     color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Feature Distributions":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_fraud']

        selected_col = st.selectbox("Select feature:", numeric_cols)

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Distribution', 'Box Plot'))

        fig.add_trace(
            go.Histogram(x=df[selected_col], name='Distribution', nbinsx=50),
            row=1, col=1
        )

        fig.add_trace(
            go.Box(y=df[selected_col], name='Box Plot'),
            row=1, col=2
        )

        fig.update_layout(height=400, showlegend=False, title_text=f"{selected_col} Analysis")
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        fig = px.imshow(corr_matrix,
                        labels=dict(color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
        fig.update_layout(title="Feature Correlation Heatmap", height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Fraud by Category":
        if 'merchant_category' in df.columns:
            fraud_by_cat = df.groupby('merchant_category')['is_fraud'].agg(['sum', 'count', 'mean']).reset_index()
            fraud_by_cat.columns = ['Category', 'Fraud Count', 'Total', 'Fraud Rate']

            fig = px.bar(fraud_by_cat, x='Category', y='Fraud Count',
                        title='Fraud Cases by Merchant Category',
                        color='Fraud Rate', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Amount Distribution":
        if 'transaction_amount' in df.columns:
            fig = px.histogram(df, x='transaction_amount', color='is_fraud',
                             title='Transaction Amount Distribution by Fraud Status',
                             nbins=50, barmode='overlay',
                             color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Time Series Analysis":
        if 'timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['timestamp']).dt.date
            daily_fraud = df_temp.groupby('date')['is_fraud'].agg(['sum', 'count']).reset_index()
            daily_fraud.columns = ['Date', 'Fraud Count', 'Total Transactions']

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            fig.add_trace(
                go.Scatter(x=daily_fraud['Date'], y=daily_fraud['Fraud Count'], name="Fraud Count"),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(x=daily_fraud['Date'], y=daily_fraud['Total Transactions'], name="Total Transactions"),
                secondary_y=True,
            )

            fig.update_layout(title_text="Daily Transaction and Fraud Trends")
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Fraud Count", secondary_y=False)
            fig.update_yaxes(title_text="Total Transactions", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)


def show_feature_engineering():
    st.markdown("### ⚙Feature Engineering")
    st.markdown("Create and transform features for better model performance")

    df = st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df

    if df is None:
        st.warning("Please extract data first")
        return

    st.markdown("#### Available Feature Engineering Operations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Features:**")
        create_time_features = st.checkbox("Time-based features (hour, day, month)", value=True)
        create_amount_features = st.checkbox("Amount-based features (log, bins)", value=True)
        create_velocity_features = st.checkbox("Velocity features (txn frequency)", value=True)

    with col2:
        st.markdown("**Advanced Features:**")
        create_aggregates = st.checkbox("Aggregated features (customer stats)", value=True)
        create_ratios = st.checkbox("Ratio features", value=True)
        create_interactions = st.checkbox("Interaction features", value=False)

    if st.button("⚙Create Features", type="primary"):
        with st.spinner("Engineering features..."):
            df_featured = create_features(
                df.copy(),
                time_features=create_time_features,
                amount_features=create_amount_features,
                velocity_features=create_velocity_features,
                aggregate_features=create_aggregates,
                ratio_features=create_ratios,
                interaction_features=create_interactions
            )

            st.session_state.df_featured = df_featured

            new_features = set(df_featured.columns) - set(df.columns)

            st.success(f"""
                Feature engineering completed!
                - Original features: {len(df.columns)}
                - New features created: {len(new_features)}
                - Total features: {len(df_featured.columns)}
            """)

            if len(new_features) > 0:
                st.markdown("**New Features Created:**")
                st.write(list(new_features))

    if st.session_state.df_featured is not None:
        st.markdown("---")
        st.markdown("#### Feature Analysis")

        df_featured = st.session_state.df_featured

        # Feature importance preview
        if 'is_fraud' in df_featured.columns:
            st.markdown("**Feature Importance (Preview):**")

            numeric_features = df_featured.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col != 'is_fraud']

            if len(numeric_features) > 0:
                # Calculate correlation with target
                correlations = df_featured[numeric_features].corrwith(df_featured['is_fraud']).abs().sort_values(ascending=False)

                fig = px.bar(x=correlations.head(15).values,
                           y=correlations.head(15).index,
                           orientation='h',
                           title='Top 15 Features by Correlation with Fraud',
                           labels={'x': 'Absolute Correlation', 'y': 'Feature'})
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Featured Data Preview")
        st.dataframe(df_featured.head(20), use_container_width=True)


def show_feature_selection():
    st.markdown("### Feature Selection")
    st.markdown("Select the most important features for modeling")

    df = st.session_state.df_featured if st.session_state.df_featured is not None else \
         (st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df)

    if df is None:
        st.warning("Please extract data first")
        return

    st.markdown("#### Feature Selection Methods")

    method = st.selectbox(
        "Select method:",
        ["Correlation-based", "Mutual Information", "Chi-squared", "Random Forest Importance"]
    )

    n_features = st.slider("Number of features to select:", 5, 50, 20)

    if st.button("Select Features", type="primary"):
        with st.spinner("Selecting features..."):
            numeric_features = df.select_dtypes(include=[np.number]).columns
            numeric_features = [col for col in numeric_features if col != 'is_fraud']

            X = df[numeric_features].fillna(0)
            y = df['is_fraud']

            if method == "Correlation-based":
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected = correlations.head(n_features).index.tolist()
                scores = correlations.head(n_features).values

            elif method == "Mutual Information":
                mi_scores = mutual_info_classif(X, y, random_state=42)
                mi_df = pd.DataFrame({'feature': numeric_features, 'score': mi_scores})
                mi_df = mi_df.sort_values('score', ascending=False)
                selected = mi_df.head(n_features)['feature'].tolist()
                scores = mi_df.head(n_features)['score'].values

            elif method == "Chi-squared":
                # Make data non-negative for chi2
                X_pos = X - X.min() + 1
                chi_scores = SelectKBest(chi2, k=min(n_features, len(numeric_features))).fit(X_pos, y)
                chi_df = pd.DataFrame({'feature': numeric_features, 'score': chi_scores.scores_})
                chi_df = chi_df.sort_values('score', ascending=False)
                selected = chi_df.head(n_features)['feature'].tolist()
                scores = chi_df.head(n_features)['score'].values

            else:  # Random Forest Importance
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                importance_df = pd.DataFrame({
                    'feature': numeric_features,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=False)
                selected = importance_df.head(n_features)['feature'].tolist()
                scores = importance_df.head(n_features)['importance'].values

            st.session_state.selected_features = selected

            st.success(f"Selected {len(selected)} features using {method}")

            # Visualization
            fig = px.bar(x=scores, y=selected, orientation='h',
                        title=f'Feature Scores ({method})',
                        labels={'x': 'Score', 'y': 'Feature'})
            st.plotly_chart(fig, use_container_width=True)

            # Display selected features
            st.markdown("**Selected Features:**")
            st.write(selected)


def show_customer_profiling():
    st.markdown("### 👥 Customer Profiling")
    st.markdown("Analyze customer behavior and segment profiles")

    df = st.session_state.df_featured if st.session_state.df_featured is not None else \
         (st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df)

    if df is None:
        st.warning("Please extract data first")
        return

    # Customer-level aggregation
    if 'customer_id' in df.columns:
        customer_profile = df.groupby('customer_id').agg({
            'transaction_amount': ['count', 'sum', 'mean', 'std', 'max'],
            'is_fraud': ['sum', 'mean']
        }).reset_index()

        customer_profile.columns = ['customer_id', 'txn_count', 'total_amount', 'avg_amount',
                                    'std_amount', 'max_amount', 'fraud_count', 'fraud_rate']

        # Customer segments
        st.markdown("#### Customer Segments")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            high_value = customer_profile[customer_profile['total_amount'] > customer_profile['total_amount'].quantile(0.9)]
            st.metric("High Value Customers", f"{len(high_value):,}")

        with col2:
            high_risk = customer_profile[customer_profile['fraud_rate'] > 0.1]
            st.metric("High Risk Customers", f"{len(high_risk):,}")

        with col3:
            frequent = customer_profile[customer_profile['txn_count'] > customer_profile['txn_count'].quantile(0.9)]
            st.metric("Frequent Customers", f"{len(frequent):,}")

        with col4:
            st.metric("Total Customers", f"{len(customer_profile):,}")

        # Visualizations
        st.markdown("---")
        viz_option = st.selectbox(
            "Select visualization:",
            ["Transaction Count Distribution", "Amount Distribution",
             "Fraud Rate by Segment", "Customer Scatter Plot"]
        )

        if viz_option == "Transaction Count Distribution":
            fig = px.histogram(customer_profile, x='txn_count',
                             title='Distribution of Transaction Count per Customer',
                             nbins=50)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Amount Distribution":
            fig = px.histogram(customer_profile, x='total_amount',
                             title='Distribution of Total Transaction Amount per Customer',
                             nbins=50)
            st.plotly_chart(fig, use_container_width=True)

        elif viz_option == "Fraud Rate by Segment":
            # Create segments
            customer_profile['segment'] = pd.cut(customer_profile['total_amount'],
                                                 bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            segment_fraud = customer_profile.groupby('segment')['fraud_rate'].mean().reset_index()

            fig = px.bar(segment_fraud, x='segment', y='fraud_rate',
                        title='Average Fraud Rate by Customer Segment',
                        labels={'fraud_rate': 'Fraud Rate', 'segment': 'Spending Segment'})
            st.plotly_chart(fig, use_container_width=True)

        else:  # Scatter Plot
            fig = px.scatter(customer_profile, x='txn_count', y='total_amount',
                           color='fraud_rate', size='fraud_count',
                           title='Customer Profile Scatter Plot',
                           labels={'txn_count': 'Transaction Count',
                                  'total_amount': 'Total Amount',
                                  'fraud_rate': 'Fraud Rate'},
                           color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Customer Profile Data")
        st.dataframe(customer_profile.head(100), use_container_width=True)

    else:
        st.info("Customer profiling requires 'customer_id' field in the dataset")


def show_train_test_split():
    st.markdown("### 🔀 Train/Test Split")
    st.markdown("Split data into training and testing sets")

    df = st.session_state.df_featured if st.session_state.df_featured is not None else \
         (st.session_state.df_cleaned if st.session_state.df_cleaned is not None else st.session_state.df)

    if df is None:
        st.warning("Please extract data first")
        return

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
        random_state = st.number_input("Random seed:", min_value=0, max_value=1000, value=42)

    with col2:
        stratify = st.checkbox("Stratified split (maintain fraud ratio)", value=True)
        shuffle = st.checkbox("Shuffle data", value=True)

    # Feature selection
    st.markdown("#### Select Features for Modeling")

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraud' in numeric_features:
        numeric_features.remove('is_fraud')

    if st.session_state.selected_features:
        default_features = [f for f in st.session_state.selected_features if f in numeric_features]
    else:
        default_features = numeric_features[:20] if len(numeric_features) > 20 else numeric_features

    selected_cols = st.multiselect(
        "Select features for modeling:",
        options=numeric_features,
        default=default_features
    )

    if st.button("🔀 Split Data", type="primary"):
        if len(selected_cols) == 0:
            st.error("Please select at least one feature")
            return

        with st.spinner("Splitting data..."):
            X = df[selected_cols].fillna(0)
            y = df['is_fraud']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if stratify else None,
                shuffle=shuffle
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Store in session state
            st.session_state.X_train = pd.DataFrame(X_train_scaled, columns=selected_cols)
            st.session_state.X_test = pd.DataFrame(X_test_scaled, columns=selected_cols)
            st.session_state.y_train = y_train.reset_index(drop=True)
            st.session_state.y_test = y_test.reset_index(drop=True)
            st.session_state.feature_columns = selected_cols

            st.success("Data split completed!")

    if st.session_state.X_train is not None:
        st.markdown("---")
        st.markdown("#### Split Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Training Samples", f"{len(st.session_state.X_train):,}")
        with col2:
            st.metric("Test Samples", f"{len(st.session_state.X_test):,}")
        with col3:
            train_fraud = st.session_state.y_train.sum()
            st.metric("Train Fraud Cases", f"{train_fraud:,}")
        with col4:
            test_fraud = st.session_state.y_test.sum()
            st.metric("Test Fraud Cases", f"{test_fraud:,}")

        # Distribution comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Train', x=['Non-Fraud', 'Fraud'],
                            y=[(st.session_state.y_train == 0).sum(),
                               (st.session_state.y_train == 1).sum()]))
        fig.add_trace(go.Bar(name='Test', x=['Non-Fraud', 'Fraud'],
                            y=[(st.session_state.y_test == 0).sum(),
                               (st.session_state.y_test == 1).sum()]))
        fig.update_layout(title='Train/Test Distribution', barmode='group')
        st.plotly_chart(fig, use_container_width=True)


def show_model_building():
    st.markdown("### Model Building & Training")
    st.markdown("Build and train machine learning models")

    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning("Please split the data first in the 'Train/Test Split' tab")
        return

    st.markdown("#### Select Models to Train")

    col1, col2, col3 = st.columns(3)

    with col1:
        train_lr = st.checkbox("Logistic Regression", value=True)
        if train_lr:
            lr_c = st.slider("LR Regularization (C):", 0.001, 10.0, 1.0)

    with col2:
        train_rf = st.checkbox("Random Forest", value=True)
        if train_rf:
            rf_estimators = st.slider("RF Trees:", 50, 500, 100, 50)

    with col3:
        train_gb = st.checkbox("Gradient Boosting", value=True)
        if train_gb:
            gb_estimators = st.slider("GB Trees:", 50, 500, 100, 50)

    if st.button("Train Models", type="primary"):
        if not (train_lr or train_rf or train_gb):
            st.error("Please select at least one model")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        models_trained = 0
        total_models = sum([train_lr, train_rf, train_gb])

        if train_lr:
            status_text.text("Training Logistic Regression...")
            lr_model = LogisticRegression(C=lr_c, random_state=42, max_iter=1000)
            lr_model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.models['Logistic Regression'] = lr_model
            models_trained += 1
            progress_bar.progress(models_trained / total_models)

        if train_rf:
            status_text.text("Training Random Forest...")
            rf_model = RandomForestClassifier(n_estimators=rf_estimators, random_state=42, n_jobs=-1)
            rf_model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.models['Random Forest'] = rf_model
            models_trained += 1
            progress_bar.progress(models_trained / total_models)

        if train_gb:
            status_text.text("Training Gradient Boosting...")
            gb_model = GradientBoostingClassifier(n_estimators=gb_estimators, random_state=42)
            gb_model.fit(st.session_state.X_train, st.session_state.y_train)
            st.session_state.models['Gradient Boosting'] = gb_model
            models_trained += 1
            progress_bar.progress(models_trained / total_models)

        status_text.text("Training completed!")
        st.success(f"Successfully trained {len(st.session_state.models)} model(s)!")

    if len(st.session_state.models) > 0:
        st.markdown("---")
        st.markdown("#### 📋 Trained Models")

        for model_name, model in st.session_state.models.items():
            with st.expander(f"{model_name}"):
                col1, col2 = st.columns(2)

                with col1:
                    # Training predictions
                    y_train_pred = model.predict(st.session_state.X_train)
                    train_acc = (y_train_pred == st.session_state.y_train).mean()
                    st.metric("Training Accuracy", f"{train_acc:.4f}")

                with col2:
                    # Test predictions
                    y_test_pred = model.predict(st.session_state.X_test)
                    test_acc = (y_test_pred == st.session_state.y_test).mean()
                    st.metric("Test Accuracy", f"{test_acc:.4f}")

                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': st.session_state.feature_columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(15)

                    fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                                title=f'Top 15 Feature Importances - {model_name}')
                    st.plotly_chart(fig, use_container_width=True)


def show_model_evaluation():
    st.markdown("### Model Evaluation")
    st.markdown("Evaluate model performance with detailed metrics")

    if len(st.session_state.models) == 0:
        st.warning("Please train models first in the 'Model Building' tab")
        return

    # Select model to evaluate
    model_name = st.selectbox("Select model to evaluate:", list(st.session_state.models.keys()))
    model = st.session_state.models[model_name]

    # Make predictions
    y_pred = model.predict(st.session_state.X_test)
    y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Metrics
    st.markdown("#### Classification Metrics")

    col1, col2, col3, col4 = st.columns(4)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    with col1:
        acc = accuracy_score(st.session_state.y_test, y_pred)
        st.metric("Accuracy", f"{acc:.4f}")

    with col2:
        prec = precision_score(st.session_state.y_test, y_pred)
        st.metric("Precision", f"{prec:.4f}")

    with col3:
        rec = recall_score(st.session_state.y_test, y_pred)
        st.metric("Recall", f"{rec:.4f}")

    with col4:
        f1 = f1_score(st.session_state.y_test, y_pred)
        st.metric("F1-Score", f"{f1:.4f}")

    # Confusion Matrix
    st.markdown("---")
    st.markdown("#### Confusion Matrix")

    cm = confusion_matrix(st.session_state.y_test, y_pred)

    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Non-Fraud', 'Fraud'],
                    y=['Non-Fraud', 'Fraud'],
                    text_auto=True,
                    color_continuous_scale='Blues')
    fig.update_layout(title=f'Confusion Matrix - {model_name}')
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    if y_pred_proba is not None:
        st.markdown("---")
        st.markdown("#### ROC Curve")

        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_pred_proba)
        auc = roc_auc_score(st.session_state.y_test, y_pred_proba)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC = {auc:.4f})',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title=f'ROC Curve - {model_name}',
                         xaxis_title='False Positive Rate',
                         yaxis_title='True Positive Rate')
        st.plotly_chart(fig, use_container_width=True)

        st.metric("ROC-AUC Score", f"{auc:.4f}")

    # Classification Report
    st.markdown("---")
    st.markdown("#### 📋 Detailed Classification Report")

    report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)


def show_final_metrics():
    st.markdown("### 📋 Final Metrics Summary")
    st.markdown("Comprehensive overview of all models and results")

    if len(st.session_state.models) == 0:
        st.warning("Please train models first")
        return

    # Collect metrics for all models
    metrics_data = []

    for model_name, model in st.session_state.models.items():
        y_pred = model.predict(st.session_state.X_test)
        y_pred_proba = model.predict_proba(st.session_state.X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(st.session_state.y_test, y_pred),
            'Precision': precision_score(st.session_state.y_test, y_pred),
            'Recall': recall_score(st.session_state.y_test, y_pred),
            'F1-Score': f1_score(st.session_state.y_test, y_pred),
        }

        if y_pred_proba is not None:
            metrics['ROC-AUC'] = roc_auc_score(st.session_state.y_test, y_pred_proba)

        metrics_data.append(metrics)

    metrics_df = pd.DataFrame(metrics_data)

    # Display metrics table
    st.markdown("#### Model Comparison")
    st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']),
                use_container_width=True)

    # Visualize comparison
    st.markdown("---")
    st.markdown("#### Performance Comparison")

    metric_to_plot = st.selectbox("Select metric:", ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])

    if metric_to_plot in metrics_df.columns:
        fig = px.bar(metrics_df, x='Model', y=metric_to_plot,
                    title=f'{metric_to_plot} Comparison Across Models',
                    color=metric_to_plot, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    # Best model recommendation
    st.markdown("---")
    st.markdown("#### Best Model Recommendation")

    if 'F1-Score' in metrics_df.columns:
        best_model_idx = metrics_df['F1-Score'].idxmax()
        best_model = metrics_df.loc[best_model_idx, 'Model']
        best_f1 = metrics_df.loc[best_model_idx, 'F1-Score']

        st.success(f"""
            **Recommended Model: {best_model}**
            - F1-Score: {best_f1:.4f}
            - Best balance between precision and recall
        """)

    # Dataset summary
    st.markdown("---")
    st.markdown("#### Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state.df is not None:
            st.metric("Total Records", f"{len(st.session_state.df):,}")

    with col2:
        if st.session_state.X_train is not None:
            st.metric("Training Set", f"{len(st.session_state.X_train):,}")

    with col3:
        if st.session_state.X_test is not None:
            st.metric("Test Set", f"{len(st.session_state.X_test):,}")

    with col4:
        if st.session_state.df_featured is not None:
            st.metric("Total Features", len(st.session_state.df_featured.columns))
