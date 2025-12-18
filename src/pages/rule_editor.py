"""
Rule Editor & Fraud Monitoring Platform
Create, edit, apply and test rules for fraud detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sys
import os
sys.path.append('/home/user/ArgusAI')
from src.utils.data_generator import generate_fraud_data
from src.utils.rule_engine import RuleEngine, Rule
from src.utils.sql_rule_loader import SQLRuleLoader, create_sample_rules


def show():
    st.markdown('<p class="main-header">📋 Rule Editor & Fraud Monitoring</p>', unsafe_allow_html=True)
    st.markdown("Create, manage, and test fraud detection rules")

    # Initialize session state
    if 'rules' not in st.session_state:
        st.session_state.rules = initialize_default_rules()
    if 'rule_results' not in st.session_state:
        st.session_state.rule_results = None
    if 'monitoring_data' not in st.session_state:
        st.session_state.monitoring_data = None
    if 'sql_rules' not in st.session_state:
        st.session_state.sql_rules = None
    if 'sql_rule_results' not in st.session_state:
        st.session_state.sql_rule_results = None

    # Create tabs
    tabs = st.tabs(["Rule Manager", "SQL Rules", "Rule Testing", "Monitoring Dashboard", "Rule Performance"])

    with tabs[0]:
        show_rule_manager()

    with tabs[1]:
        show_sql_rules()

    with tabs[2]:
        show_rule_testing()

    with tabs[3]:
        show_monitoring_dashboard()

    with tabs[4]:
        show_rule_performance()


def initialize_default_rules():
    """Initialize with some default fraud detection rules"""
    default_rules = [
        {
            'id': 'RULE_001',
            'name': 'High Amount Transaction',
            'description': 'Flag transactions above threshold amount',
            'sql_template': 'transaction_amount > {threshold}',
            'threshold': 5000,
            'priority': 'High',
            'status': 'Active',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'actions': ['Flag', 'Alert']
        },
        {
            'id': 'RULE_002',
            'name': 'Multiple Transactions Same Merchant',
            'description': 'Multiple transactions at same merchant in short time',
            'sql_template': 'merchant_velocity > {threshold}',
            'threshold': 3,
            'priority': 'Medium',
            'status': 'Active',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'actions': ['Flag']
        },
        {
            'id': 'RULE_003',
            'name': 'Foreign Transaction',
            'description': 'Transaction in foreign country',
            'sql_template': "country != '{home_country}'",
            'threshold': 'USA',
            'priority': 'Medium',
            'status': 'Active',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'actions': ['Review']
        },
        {
            'id': 'RULE_004',
            'name': 'High Risk Merchant Category',
            'description': 'Transaction in high-risk merchant category',
            'sql_template': "merchant_category IN {categories}",
            'threshold': "('gambling', 'crypto', 'wire_transfer')",
            'priority': 'High',
            'status': 'Active',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'actions': ['Flag', 'Review']
        },
        {
            'id': 'RULE_005',
            'name': 'Unusual Transaction Hour',
            'description': 'Transaction during unusual hours',
            'sql_template': 'hour < {start_hour} OR hour > {end_hour}',
            'threshold': '6,22',
            'priority': 'Low',
            'status': 'Active',
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'actions': ['Flag']
        }
    ]
    return default_rules


def show_rule_manager():
    st.markdown("### Rule Manager")
    st.markdown("Create, edit, and manage fraud detection rules")

    # Load from Rules Directory section
    st.markdown("---")
    st.markdown("### 📁 Load Rules from Directory")


    # Initialize rules directory in session state
    if 'rules_directory' not in st.session_state:
        st.session_state.rules_directory = "/root/@dfs-ai-app2/rules"

    col1, col2 = st.columns([3, 1])
    with col1:
        rules_dir = st.text_input(
            "Rules Directory Path:",
            value=st.session_state.rules_directory,
            help="Path to directory containing rule JSON files",
            key="rules_dir_input"
        )
    with col2:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="refresh_rules_dir"):
            st.session_state.rules_directory = rules_dir
            st.rerun()

    st.info(f"📁 Current directory: `{rules_dir}`")

    # Scan for rule files
    import os
    import glob

    if os.path.exists(rules_dir) and os.path.isdir(rules_dir):
        rule_files = glob.glob(os.path.join(rules_dir, "*.json"))

        if len(rule_files) > 0:
            st.success(f"Found {len(rule_files)} rule file(s)")

            # Display rule files
            for rule_file in sorted(rule_files, reverse=True):
                rule_filename = os.path.basename(rule_file)

                with st.expander(f"📄 {rule_filename}", expanded=False):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.markdown("**File Information:**")
                        file_size = os.path.getsize(rule_file)
                        st.text(f"Size: {file_size} bytes")

                        mod_time = datetime.fromtimestamp(os.path.getmtime(rule_file))
                        st.text(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")

                    with col2:
                        st.markdown("**Preview:**")
                        try:
                            with open(rule_file, 'r') as f:
                                rule_data = json.load(f)

                            # Handle both single rule and array of rules
                            if isinstance(rule_data, list):
                                st.caption(f"Contains {len(rule_data)} rule(s)")
                                if len(rule_data) > 0:
                                    st.caption(f"First rule: {rule_data[0].get('name', 'Unnamed')}")
                            else:
                                st.caption(f"Rule: {rule_data.get('name', 'Unnamed')}")
                                st.caption(f"Priority: {rule_data.get('priority', 'N/A')}")
                        except Exception as e:
                            st.caption(f"Error reading: {str(e)}")

                    with col3:
                        st.markdown("**Actions:**")
                        if st.button("Load", key=f"load_rule_{rule_filename}"):
                            try:
                                with open(rule_file, 'r') as f:
                                    rule_data = json.load(f)

                                # Handle both single rule and array of rules
                                if isinstance(rule_data, list):
                                    for rule in rule_data:
                                        # Check if rule already exists
                                        existing_ids = [r['id'] for r in st.session_state.rules]
                                        if rule.get('id') not in existing_ids:
                                            st.session_state.rules.append(rule)
                                    st.success(f"Loaded {len(rule_data)} rule(s) from {rule_filename}")
                                else:
                                    # Single rule
                                    existing_ids = [r['id'] for r in st.session_state.rules]
                                    if rule_data.get('id') not in existing_ids:
                                        st.session_state.rules.append(rule_data)
                                        st.success(f"Loaded rule from {rule_filename}")
                                    else:
                                        st.warning(f"Rule {rule_data.get('id')} already exists")

                                st.rerun()
                            except Exception as e:
                                st.error(f"Error loading rule: {str(e)}")

                        if st.button("View JSON", key=f"view_rule_{rule_filename}"):
                            try:
                                with open(rule_file, 'r') as f:
                                    rule_json = f.read()
                                st.code(rule_json, language='json')
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")

            # Bulk load all rules
            st.markdown("---")
            if st.button("📥 Load All Rules from Directory", type="primary"):
                loaded_count = 0
                error_count = 0

                for rule_file in rule_files:
                    try:
                        with open(rule_file, 'r') as f:
                            rule_data = json.load(f)

                        if isinstance(rule_data, list):
                            for rule in rule_data:
                                existing_ids = [r['id'] for r in st.session_state.rules]
                                if rule.get('id') not in existing_ids:
                                    st.session_state.rules.append(rule)
                                    loaded_count += 1
                        else:
                            existing_ids = [r['id'] for r in st.session_state.rules]
                            if rule_data.get('id') not in existing_ids:
                                st.session_state.rules.append(rule_data)
                                loaded_count += 1
                    except Exception as e:
                        error_count += 1

                if loaded_count > 0:
                    st.success(f"✅ Loaded {loaded_count} new rule(s) from directory")
                if error_count > 0:
                    st.warning(f"⚠️ Failed to load {error_count} file(s)")
                st.rerun()
        else:
            st.info("No rule files (.json) found in directory")
            st.markdown("**Tip:** Create rules using the editor below and export them to this directory")
    else:
        st.warning(f"Directory `{rules_dir}` does not exist")
        st.markdown("**Tip:** Create the directory or specify an existing path")

    # Export current rules to directory
    st.markdown("---")
    st.markdown("#### Export Rules to Directory")

    col1, col2 = st.columns([2, 1])
    with col1:
        export_filename = st.text_input(
            "Export Filename:",
            value=f"rules_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            key="export_filename"
        )
    with col2:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("💾 Export All Rules", key="export_rules"):
            try:
                # Create directory if it doesn't exist
                os.makedirs(rules_dir, exist_ok=True)

                export_path = os.path.join(rules_dir, export_filename)
                with open(export_path, 'w') as f:
                    json.dump(st.session_state.rules, f, indent=2)

                st.success(f"✅ Exported {len(st.session_state.rules)} rule(s) to {export_path}")
            except Exception as e:
                st.error(f"Error exporting rules: {str(e)}")

    # Add new rule section
    with st.expander("➕ Create New Rule", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            new_rule_id = st.text_input("Rule ID:", value=f"RULE_{len(st.session_state.rules) + 1:03d}")
            new_rule_name = st.text_input("Rule Name:")
            new_rule_desc = st.text_area("Description:")

        with col2:
            new_rule_priority = st.selectbox("Priority:", ["Low", "Medium", "High", "Critical"])
            new_rule_status = st.selectbox("Status:", ["Active", "Inactive", "Testing"])
            new_rule_actions = st.multiselect("Actions:", ["Flag", "Alert", "Review", "Block", "Notify"])

        st.markdown("#### Rule Configuration")
        col1, col2 = st.columns(2)

        with col1:
            rule_type = st.selectbox(
                "Rule Type:",
                ["Amount Threshold", "Velocity Check", "Geographic", "Merchant Category",
                 "Time-based", "Custom SQL"]
            )

        with col2:
            if rule_type == "Amount Threshold":
                threshold = st.number_input("Amount Threshold:", value=5000.0)
                sql_template = f"transaction_amount > {threshold}"
            elif rule_type == "Velocity Check":
                threshold = st.number_input("Transaction Count:", value=3)
                time_window = st.selectbox("Time Window:", ["1 hour", "24 hours", "7 days"])
                sql_template = f"transaction_count_{time_window.replace(' ', '_')} > {threshold}"
            elif rule_type == "Geographic":
                countries = st.text_input("Excluded Countries (comma-separated):", value="USA,UK")
                sql_template = f"country NOT IN ('{countries}')"
            elif rule_type == "Merchant Category":
                categories = st.text_input("High-Risk Categories (comma-separated):", value="gambling,crypto")
                sql_template = f"merchant_category IN ('{categories}')"
            elif rule_type == "Time-based":
                start_hour = st.number_input("Start Hour:", 0, 23, 6)
                end_hour = st.number_input("End Hour:", 0, 23, 22)
                sql_template = f"hour < {start_hour} OR hour > {end_hour}"
            else:
                sql_template = st.text_area("Custom SQL Template:", value="transaction_amount > {threshold}")
                threshold = st.text_input("Threshold/Parameter:", value="5000")

        st.markdown("**Generated SQL Template:**")
        st.code(sql_template, language='sql')

        if st.button("Save Rule", type="primary"):
            if new_rule_name and new_rule_id:
                new_rule = {
                    'id': new_rule_id,
                    'name': new_rule_name,
                    'description': new_rule_desc,
                    'sql_template': sql_template,
                    'threshold': threshold if rule_type != "Custom SQL" else threshold,
                    'priority': new_rule_priority,
                    'status': new_rule_status,
                    'created_date': datetime.now().strftime('%Y-%m-%d'),
                    'actions': new_rule_actions
                }
                st.session_state.rules.append(new_rule)
                st.success(f"Rule '{new_rule_name}' created successfully!")
            else:
                st.error("Please provide Rule ID and Name")

    # Display existing rules
    st.markdown("---")
    st.markdown("### Existing Rules")

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_status = st.multiselect("Filter by Status:", ["Active", "Inactive", "Testing"],
                                       default=["Active", "Testing"])
    with col2:
        filter_priority = st.multiselect("Filter by Priority:", ["Low", "Medium", "High", "Critical"],
                                        default=["Low", "Medium", "High", "Critical"])
    with col3:
        search_term = st.text_input("Search rules:")

    # Filter rules
    filtered_rules = [r for r in st.session_state.rules
                     if r['status'] in filter_status
                     and r['priority'] in filter_priority
                     and (not search_term or search_term.lower() in r['name'].lower()
                          or search_term.lower() in r['description'].lower())]

    # Display rules as cards
    for i, rule in enumerate(filtered_rules):
        with st.expander(f"🔖 {rule['id']}: {rule['name']} [{rule['status']}]"):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown(f"**Description:** {rule['description']}")
                st.markdown(f"**SQL Template:**")
                st.code(rule['sql_template'], language='sql')
                st.markdown(f"**Threshold:** {rule['threshold']}")
                st.markdown(f"**Actions:** {', '.join(rule['actions'])}")

            with col2:
                st.markdown(f"**Priority:** {rule['priority']}")
                st.markdown(f"**Status:** {rule['status']}")
                st.markdown(f"**Created:** {rule['created_date']}")

            # Rule actions
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button("Edit", key=f"edit_{i}"):
                    st.info("Edit functionality - would open edit dialog")

            with col2:
                if st.button("Test", key=f"test_{i}"):
                    st.info("Test functionality - would run rule on sample data")

            with col3:
                new_status = st.selectbox("Change Status:", ["Active", "Inactive", "Testing"],
                                         index=["Active", "Inactive", "Testing"].index(rule['status']),
                                         key=f"status_{i}")
                if new_status != rule['status']:
                    rule['status'] = new_status
                    st.success(f"Status updated to {new_status}")

            with col4:
                if st.button("Delete", key=f"delete_{i}"):
                    st.session_state.rules.remove(rule)
                    st.rerun()

    # Summary metrics
    st.markdown("---")
    st.markdown("### Rule Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rules", len(st.session_state.rules))

    with col2:
        active_rules = len([r for r in st.session_state.rules if r['status'] == 'Active'])
        st.metric("Active Rules", active_rules)

    with col3:
        high_priority = len([r for r in st.session_state.rules if r['priority'] in ['High', 'Critical']])
        st.metric("High Priority", high_priority)

    with col4:
        testing_rules = len([r for r in st.session_state.rules if r['status'] == 'Testing'])
        st.metric("Testing", testing_rules)




def show_sql_rules():
    """SQL-Based Rule Management - Load and execute SQL rules from files"""
    st.markdown("### SQL-Based Fraud Detection Rules")
    st.markdown("Load SQL WHERE clause rules from files and execute on loaded data")

    # Initialize SQL rule loader
    rules_dir = "/root/research-dir/dev/jazzcash-fraud-detection/rules"

    # Rules directory configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_dir = st.text_input(
            "Rules Directory:",
            value=rules_dir,
            help="Directory containing SQL rule files (.sql or .txt)"
        )
    with col2:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", key="refresh_sql_rules"):
            st.session_state.sql_rules = None
            st.rerun()

    # Check if directory exists
    if not os.path.exists(custom_dir):
        st.warning(f"⚠️ Directory does not exist: {custom_dir}")
        st.info("💡 Create sample rules to get started:")

        if st.button("📝 Create Sample SQL Rules"):
            try:
                create_sample_rules(custom_dir)
                st.success(f"✅ Created sample SQL rule files in {custom_dir}")
                st.info("Sample rules include: high amounts, velocity checks, unusual patterns")
                st.rerun()
            except Exception as e:
                st.error(f"Error creating sample rules: {str(e)}")
        return

    # Load SQL rules
    st.markdown("---")
    st.markdown("#### Load SQL Rules")

    loader = SQLRuleLoader(custom_dir)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("📥 Load All SQL Rules", type="primary"):
            try:
                with st.spinner("Loading SQL rules..."):
                    rules = loader.load_rules_from_directory()
                    st.session_state.sql_rules = loader
                    st.success(f"✅ Loaded {len(rules)} SQL rule(s)")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading rules: {str(e)}")

    with col2:
        if st.button("📝 Create Samples"):
            try:
                create_sample_rules(custom_dir)
                st.success("✅ Sample rules created")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

    with col3:
        if st.button("🗑️ Clear Loaded"):
            st.session_state.sql_rules = None
            st.session_state.sql_rule_results = None
            st.rerun()

    # Display loaded rules
    if st.session_state.sql_rules is not None:
        loader = st.session_state.sql_rules
        st.markdown("---")
        st.markdown("#### Loaded SQL Rules")

        if len(loader.rules) == 0:
            st.info("No rules loaded. Load rules from directory or create samples.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rules", len(loader.rules))
        with col2:
            high_pri = len([r for r in loader.rules.values() if r.priority in ['High', 'Critical']])
            st.metric("High Priority", high_pri)
        with col3:
            medium_pri = len([r for r in loader.rules.values() if r.priority == 'Medium'])
            st.metric("Medium Priority", medium_pri)
        with col4:
            low_pri = len([r for r in loader.rules.values() if r.priority == 'Low'])
            st.metric("Low Priority", low_pri)

        # Display rule summary
        st.markdown("**Rule Summary:**")
        summary_df = loader.get_rule_summary()
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Execute rules on loaded data
        st.markdown("---")
        st.markdown("#### Execute Rules on Data")

        if st.session_state.loaded_data is None:
            st.warning("⚠️ No data loaded.")
            st.info("💡 Go to **Data Loading** module to load transaction data first.")
            return

        data = st.session_state.loaded_data
        st.info(f"📊 Loaded data: {len(data):,} rows")

        # Check for fraud_flag column
        if 'fraud_flag' not in data.columns:
            st.warning("⚠️ No fraud_flag column in data. Performance metrics will not be available.")
            target_col = None
        else:
            target_col = 'fraud_flag'
            fraud_count = data[target_col].sum()
            st.info(f"🔍 Total fraud cases in data: {fraud_count:,} ({fraud_count/len(data)*100:.2f}%)")

        if st.button("▶️ Execute All SQL Rules", type="primary"):
            try:
                with st.spinner(f"Executing {len(loader.rules)} rules..."):
                    results = loader.execute_rules(data, target_col or 'fraud_flag')
                    st.session_state.sql_rule_results = results
                    st.success(f"✅ Executed {len(results)} rule(s)")
                    st.rerun()
            except Exception as e:
                st.error(f"Error executing rules: {str(e)}")
                st.exception(e)

        # Display results
        if st.session_state.sql_rule_results is not None:
            st.markdown("---")
            st.markdown("#### Rule Execution Results")

            results = st.session_state.sql_rule_results

            # Overall statistics
            total_flags = sum([r['triggered_count'] for r in results.values()])
            st.metric("Total Flags Raised", f"{total_flags:,}")

            # Performance metrics table
            st.markdown("**Rule Performance:**")

            perf_data = []
            for rule_id, result in results.items():
                rule = result['rule']
                metrics = result.get('metrics', {})

                perf_data.append({
                    'Rule ID': rule.rule_id,
                    'Rule Name': rule.name,
                    'Priority': rule.priority,
                    'Triggered': result['triggered_count'],
                    'Detection Rate %': metrics.get('detection_rate', 0),
                    'Precision %': metrics.get('precision', 0),
                    'Recall %': metrics.get('recall', 0),
                    'F1 Score': metrics.get('f1_score', 0),
                    'True Positives': metrics.get('true_positives', 0),
                    'False Positives': metrics.get('false_positives', 0)
                })

            perf_df = pd.DataFrame(perf_data)

            # Color code by performance
            def color_performance(val):
                if isinstance(val, (int, float)):
                    if val >= 80:
                        return 'background-color: #d4edda'
                    elif val >= 60:
                        return 'background-color: #fff3cd'
                    elif val > 0:
                        return 'background-color: #f8d7da'
                return ''

            styled_df = perf_df.style.applymap(
                color_performance,
                subset=['Detection Rate %', 'Precision %', 'Recall %']
            )

            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Visualization
            st.markdown("---")
            st.markdown("**Rule Performance Comparison:**")

            col1, col2 = st.columns(2)

            with col1:
                # Detection rate chart
                fig = px.bar(
                    perf_df,
                    x='Rule Name',
                    y='Detection Rate %',
                    color='Priority',
                    title='Detection Rate by Rule',
                    color_discrete_map={'Low': '#17a2b8', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Precision vs Recall
                fig = px.scatter(
                    perf_df,
                    x='Recall %',
                    y='Precision %',
                    size='Triggered',
                    color='Priority',
                    hover_data=['Rule Name'],
                    title='Precision vs Recall',
                    color_discrete_map={'Low': '#17a2b8', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed results per rule
            st.markdown("---")
            st.markdown("**Detailed Rule Results:**")

            for rule_id, result in results.items():
                rule = result['rule']
                metrics = result.get('metrics', {})

                with st.expander(f"📋 {rule.rule_id}: {rule.name} - Triggered {result['triggered_count']} times"):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("**Rule Details:**")
                        st.text(f"Priority: {rule.priority}")
                        st.text(f"Description: {rule.description}")
                        st.markdown("**SQL Condition:**")
                        st.code(rule.sql_condition, language='sql')

                    with col2:
                        st.markdown("**Performance Metrics:**")
                        if metrics:
                            st.metric("Detection Rate", f"{metrics.get('detection_rate', 0):.2f}%")
                            st.metric("Precision", f"{metrics.get('precision', 0):.2f}%")
                            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")

                    if metrics:
                        st.markdown("**Confusion Matrix:**")
                        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
                        with cm_col1:
                            st.metric("True Positives", metrics.get('true_positives', 0))
                        with cm_col2:
                            st.metric("False Positives", metrics.get('false_positives', 0))
                        with cm_col3:
                            st.metric("False Negatives", metrics.get('false_negatives', 0))
                        with cm_col4:
                            st.metric("True Negatives", metrics.get('true_negatives', 0))

            # Export results
            st.markdown("---")
            st.markdown("**Export Results:**")

            col1, col2 = st.columns(2)
            with col1:
                csv = perf_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Performance CSV",
                    data=csv,
                    file_name=f"sql_rule_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col2:
                # Export flagged transactions
                all_flagged_indices = set()
                for result in results.values():
                    all_flagged_indices.update(result['triggered_indices'])

                if len(all_flagged_indices) > 0:
                    flagged_data = data.loc[list(all_flagged_indices)]
                    flagged_csv = flagged_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Flagged Transactions",
                        data=flagged_csv,
                        file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    else:
        st.info("👆 Click 'Load All SQL Rules' to get started")
def show_rule_testing():
    st.markdown("### Rule Testing")
    st.markdown("Test rules against sample or real data")

    if len(st.session_state.rules) == 0:
        st.warning("No rules available. Please create rules in the Rule Manager tab.")
        return

    # Data source
    st.markdown("#### 1⃣ Select Data Source")

    col1, col2 = st.columns(2)

    with col1:
        data_source = st.radio("Data Source:", ["Generate Sample Data", "Use Existing Data", "Upload CSV"])

    with col2:
        if data_source == "Generate Sample Data":
            n_samples = st.number_input("Number of records:", 100, 10000, 1000)

    # Load/Generate data
    if st.button("Load Data", type="primary"):
        with st.spinner("Loading data..."):
            if data_source == "Generate Sample Data":
                test_data = generate_fraud_data(n_samples=n_samples, fraud_rate=0.05)
                st.session_state.test_data = test_data
                st.success(f"Generated {len(test_data)} test records")
            else:
                st.info("Other data sources would be implemented here")

    if 'test_data' in st.session_state and st.session_state.test_data is not None:
        st.markdown("---")
        st.markdown("#### 2⃣ Select Rules to Test")

        # Select rules
        active_rules = [r for r in st.session_state.rules if r['status'] in ['Active', 'Testing']]

        selected_rules = st.multiselect(
            "Select rules to apply:",
            options=[f"{r['id']}: {r['name']}" for r in active_rules],
            default=[f"{r['id']}: {r['name']}" for r in active_rules[:3]]
        )

        if st.button("Run Rules", type="primary"):
            with st.spinner("Running rules..."):
                # Apply rules
                results = apply_rules_to_data(st.session_state.test_data, selected_rules, active_rules)
                st.session_state.rule_results = results

                st.success(f"Rules applied to {len(st.session_state.test_data)} records")

        # Display results
        if st.session_state.rule_results is not None:
            st.markdown("---")
            st.markdown("#### 3⃣ Test Results")

            results = st.session_state.rule_results

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Records", results['total_records'])

            with col2:
                st.metric("Flagged Records", results['flagged_records'])

            with col3:
                flagged_pct = (results['flagged_records'] / results['total_records']) * 100
                st.metric("Flagged Rate", f"{flagged_pct:.2f}%")

            with col4:
                st.metric("Rules Triggered", results['rules_triggered'])

            # Rule performance
            st.markdown("**Rule Trigger Summary:**")

            rule_summary = pd.DataFrame(results['rule_summary'])
            st.dataframe(rule_summary, use_container_width=True)

            # Visualization
            fig = px.bar(rule_summary, x='Rule', y='Triggered Count',
                        title='Rule Trigger Counts',
                        color='Triggered Count',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

            # Flagged transactions
            st.markdown("---")
            st.markdown("#### 🚩 Flagged Transactions")

            flagged_df = results['flagged_transactions']

            if len(flagged_df) > 0:
                st.dataframe(flagged_df.head(100), use_container_width=True)

                # Download button
                csv = flagged_df.to_csv(index=False)
                st.download_button(
                    label="Download Flagged Transactions",
                    data=csv,
                    file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No transactions were flagged by the selected rules")


def apply_rules_to_data(data, selected_rules, all_rules):
    """Apply selected rules to data and return results"""

    df = data.copy()
    df['flagged'] = False
    df['triggered_rules'] = ''

    # Parse selected rules
    selected_rule_ids = [r.split(':')[0].strip() for r in selected_rules]

    rule_summary = []
    total_flagged = 0
    rules_triggered = 0

    for rule in all_rules:
        if rule['id'] not in selected_rule_ids:
            continue

        # Simple rule evaluation (simplified for demo)
        triggered = evaluate_rule(df, rule)

        if triggered.sum() > 0:
            rules_triggered += 1
            df.loc[triggered, 'flagged'] = True
            df.loc[triggered, 'triggered_rules'] += f"{rule['id']}; "

        rule_summary.append({
            'Rule': rule['name'],
            'Rule ID': rule['id'],
            'Triggered Count': triggered.sum(),
            'Trigger Rate': f"{(triggered.sum() / len(df)) * 100:.2f}%"
        })

    total_flagged = df['flagged'].sum()

    results = {
        'total_records': len(df),
        'flagged_records': total_flagged,
        'rules_triggered': rules_triggered,
        'rule_summary': rule_summary,
        'flagged_transactions': df[df['flagged']]
    }

    return results


def evaluate_rule(df, rule):
    """Evaluate a rule against the dataframe"""
    try:
        # Simplified rule evaluation
        sql_template = rule['sql_template']
        threshold = rule['threshold']

        # Replace placeholders
        if 'transaction_amount' in sql_template and isinstance(threshold, (int, float)):
            if '>' in sql_template:
                return df['transaction_amount'] > threshold
            elif '<' in sql_template:
                return df['transaction_amount'] < threshold

        elif 'merchant_category' in sql_template and 'merchant_category' in df.columns:
            # Extract categories from threshold
            if isinstance(threshold, str):
                categories = [c.strip().strip("'\"") for c in threshold.strip('()').split(',')]
                return df['merchant_category'].isin(categories)

        elif 'hour' in sql_template and 'transaction_hour' in df.columns:
            # Time-based rules
            if 'OR' in sql_template:
                parts = threshold.split(',')
                if len(parts) == 2:
                    start_hour, end_hour = int(parts[0]), int(parts[1])
                    return (df['transaction_hour'] < start_hour) | (df['transaction_hour'] > end_hour)

        # Default: return False for all
        return pd.Series([False] * len(df))

    except Exception as e:
        return pd.Series([False] * len(df))


def show_monitoring_dashboard():
    st.markdown("### Real-Time Monitoring Dashboard")
    st.markdown("Monitor fraud detection in real-time")

    # Generate monitoring data if not exists
    if st.button("Generate Monitoring Data"):
        with st.spinner("Generating monitoring data..."):
            st.session_state.monitoring_data = generate_monitoring_data()
            st.success("Monitoring data generated")

    if st.session_state.monitoring_data is not None:
        data = st.session_state.monitoring_data

        # Real-time metrics
        st.markdown("#### Real-Time Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Transactions (Last Hour)", f"{data['txn_last_hour']:,}",
                     delta=f"+{data['txn_change']}%")

        with col2:
            st.metric("Flagged Cases", f"{data['flagged_cases']:,}",
                     delta=f"+{data['flagged_change']}")

        with col3:
            st.metric("Fraud Rate", f"{data['fraud_rate']:.2f}%",
                     delta=f"{data['fraud_rate_change']:.2f}%")

        with col4:
            st.metric("Avg Processing Time", f"{data['avg_processing_time']:.0f}ms")

        with col5:
            st.metric("Rules Active", data['rules_active'])

        # Time series charts
        st.markdown("---")
        st.markdown("#### Transaction Trends")

        col1, col2 = st.columns(2)

        with col1:
            # Transaction volume
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['time_series']['timestamp'],
                y=data['time_series']['total_transactions'],
                mode='lines',
                name='Total Transactions',
                line=dict(color='#744ada', width=2)
            ))
            fig.update_layout(title='Transaction Volume (Last 24 Hours)',
                            xaxis_title='Time',
                            yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Fraud cases
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['time_series']['timestamp'],
                y=data['time_series']['fraud_cases'],
                mode='lines+markers',
                name='Fraud Cases',
                line=dict(color='#000000', width=2)
            ))
            fig.update_layout(title='Fraud Cases Detected (Last 24 Hours)',
                            xaxis_title='Time',
                            yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)

        # Rule performance
        st.markdown("---")
        st.markdown("#### Rule Performance")

        rule_perf = pd.DataFrame(data['rule_performance'])
        st.dataframe(rule_perf, use_container_width=True)

        # Top flagged categories
        st.markdown("---")
        st.markdown("#### Top Alert Categories")

        col1, col2 = st.columns(2)

        with col1:
            category_data = pd.DataFrame(data['top_categories'])
            fig = px.bar(category_data, x='Count', y='Category', orientation='h',
                        title='Top Merchant Categories Flagged',
                        color='Count', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            country_data = pd.DataFrame(data['top_countries'])
            fig = px.pie(country_data, names='Country', values='Count',
                        title='Flagged Transactions by Country')
            st.plotly_chart(fig, use_container_width=True)


def generate_monitoring_data():
    """Generate sample monitoring data"""
    # Generate time series data
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')

    time_series = pd.DataFrame({
        'timestamp': hours,
        'total_transactions': np.random.randint(800, 1500, 24),
        'fraud_cases': np.random.randint(10, 50, 24)
    })

    # Rule performance
    rule_performance = [
        {'Rule': 'High Amount', 'Triggers': 145, 'True Positives': 98, 'False Positives': 47, 'Precision': 0.676},
        {'Rule': 'Velocity Check', 'Triggers': 89, 'True Positives': 67, 'False Positives': 22, 'Precision': 0.753},
        {'Rule': 'Foreign Transaction', 'Triggers': 234, 'True Positives': 156, 'False Positives': 78, 'Precision': 0.667},
        {'Rule': 'High Risk Merchant', 'Triggers': 178, 'True Positives': 142, 'False Positives': 36, 'Precision': 0.798},
        {'Rule': 'Unusual Hours', 'Triggers': 67, 'True Positives': 34, 'False Positives': 33, 'Precision': 0.507}
    ]

    # Top categories
    top_categories = [
        {'Category': 'Gambling', 'Count': 145},
        {'Category': 'Crypto', 'Count': 123},
        {'Category': 'Wire Transfer', 'Count': 98},
        {'Category': 'Electronics', 'Count': 76},
        {'Category': 'Jewelry', 'Count': 54}
    ]

    # Top countries
    top_countries = [
        {'Country': 'Nigeria', 'Count': 89},
        {'Country': 'Russia', 'Count': 67},
        {'Country': 'China', 'Count': 56},
        {'Country': 'Brazil', 'Count': 45},
        {'Country': 'India', 'Count': 34}
    ]

    return {
        'txn_last_hour': 1234,
        'txn_change': 5.2,
        'flagged_cases': 87,
        'flagged_change': 12,
        'fraud_rate': 7.05,
        'fraud_rate_change': 0.8,
        'avg_processing_time': 45.3,
        'rules_active': len([r for r in st.session_state.rules if r['status'] == 'Active']),
        'time_series': time_series,
        'rule_performance': rule_performance,
        'top_categories': top_categories,
        'top_countries': top_countries
    }


def show_rule_performance():
    st.markdown("### Rule Performance Analytics")
    st.markdown("Analyze and optimize rule effectiveness")

    if len(st.session_state.rules) == 0:
        st.warning("No rules available")
        return

    # Generate performance data
    st.markdown("#### Historical Performance")

    # Date range selector
    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date:", value=datetime.now() - timedelta(days=30))

    with col2:
        end_date = st.date_input("End Date:", value=datetime.now())

    if st.button("Generate Performance Report"):
        with st.spinner("Generating report..."):
            # Simulate performance metrics
            performance_data = []

            for rule in st.session_state.rules:
                if rule['status'] == 'Active':
                    performance_data.append({
                        'Rule ID': rule['id'],
                        'Rule Name': rule['name'],
                        'Total Triggers': np.random.randint(100, 1000),
                        'True Positives': np.random.randint(50, 800),
                        'False Positives': np.random.randint(20, 200),
                        'Precision': np.random.uniform(0.5, 0.95),
                        'Recall': np.random.uniform(0.6, 0.9),
                        'F1 Score': np.random.uniform(0.55, 0.85)
                    })

            perf_df = pd.DataFrame(performance_data)

            st.markdown("#### Performance Metrics")
            st.dataframe(perf_df.style.background_gradient(subset=['Precision', 'Recall', 'F1 Score'],
                                                          cmap='RdYlGn'),
                        use_container_width=True)

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(perf_df, x='Rule Name', y='Total Triggers',
                           title='Total Triggers by Rule',
                           color='Precision', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(perf_df, x='Precision', y='Recall',
                               size='Total Triggers', hover_data=['Rule Name'],
                               title='Precision vs Recall',
                               color='F1 Score', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)

            # Rule recommendations
            st.markdown("---")
            st.markdown("#### Recommendations")

            # Find low performing rules
            low_precision = perf_df[perf_df['Precision'] < 0.6]

            if len(low_precision) > 0:
                st.warning(f"{len(low_precision)} rule(s) with precision below 60%:")
                for _, row in low_precision.iterrows():
                    st.markdown(f"- **{row['Rule Name']}**: Precision = {row['Precision']:.2%} - Consider adjusting thresholds")

            # Find high performing rules
            high_perf = perf_df[perf_df['F1 Score'] > 0.75]

            if len(high_perf) > 0:
                st.success(f"{len(high_perf)} high-performing rule(s):")
                for _, row in high_perf.iterrows():
                    st.markdown(f"- **{row['Rule Name']}**: F1 Score = {row['F1 Score']:.2%}")
