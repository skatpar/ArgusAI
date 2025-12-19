"""
Case Management Platform
Investigate suspicious transactions in detail
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
sys.path.append('/home/user/ArgusAI')
from src.utils.data_generator import generate_fraud_data


def show():
    st.markdown('<p class="main-header">Case Management</p>', unsafe_allow_html=True)
    st.markdown("Investigate and manage suspicious transactions")

    # Initialize session state
    if 'cases' not in st.session_state:
        st.session_state.cases = generate_sample_cases()
    if 'selected_case' not in st.session_state:
        st.session_state.selected_case = None

    # Create tabs
    tabs = st.tabs(["Case Dashboard", "Case Investigation", "Case Analytics", "Case Workflow"])

    with tabs[0]:
        show_case_dashboard()

    with tabs[1]:
        show_case_investigation()

    with tabs[2]:
        show_case_analytics()

    with tabs[3]:
        show_case_workflow()


def generate_sample_cases():
    """Generate sample cases for demonstration"""
    cases = []

    statuses = ['Open', 'Under Investigation', 'Escalated', 'Closed - Fraud', 'Closed - Legitimate']
    priorities = ['Low', 'Medium', 'High', 'Critical']
    case_types = ['High Amount', 'Velocity Anomaly', 'Geographic Anomaly', 'Merchant Risk', 'Customer Behavior']

    for i in range(50):
        case_id = f"CASE_{1000 + i}"
        created_date = datetime.now() - timedelta(days=np.random.randint(0, 30))

        case = {
            'case_id': case_id,
            'transaction_id': f"TXN_{np.random.randint(100000, 999999)}",
            'customer_id': f"CUST_{np.random.randint(1000, 9999)}",
            'amount': np.random.uniform(100, 10000),
            'merchant': np.random.choice(['Amazon', 'eBay', 'Walmart', 'Best Buy', 'Unknown Merchant']),
            'status': np.random.choice(statuses),
            'priority': np.random.choice(priorities),
            'case_type': np.random.choice(case_types),
            'created_date': created_date,
            'assigned_to': np.random.choice(['Analyst A', 'Analyst B', 'Analyst C', 'Unassigned']),
            'fraud_score': np.random.uniform(0.5, 0.99),
            'triggered_rules': np.random.choice(['RULE_001', 'RULE_002, RULE_004', 'RULE_003', 'RULE_001, RULE_005']),
            'notes_count': np.random.randint(0, 10),
            'last_updated': created_date + timedelta(hours=np.random.randint(1, 48))
        }

        cases.append(case)

    return pd.DataFrame(cases)


def show_case_dashboard():
    st.markdown("### Case Dashboard")
    st.markdown("Overview of all fraud investigation cases")

    cases_df = st.session_state.cases

    # Metrics
    st.markdown("#### Key Metrics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        total_cases = len(cases_df)
        st.metric("Total Cases", total_cases)

    with col2:
        open_cases = len(cases_df[cases_df['status'].isin(['Open', 'Under Investigation'])])
        st.metric("Open Cases", open_cases)

    with col3:
        critical_cases = len(cases_df[cases_df['priority'] == 'Critical'])
        st.metric("Critical Priority", critical_cases)

    with col4:
        closed_fraud = len(cases_df[cases_df['status'] == 'Closed - Fraud'])
        fraud_rate = (closed_fraud / len(cases_df[cases_df['status'].str.startswith('Closed')])) * 100 if len(cases_df[cases_df['status'].str.startswith('Closed')]) > 0 else 0
        st.metric("Confirmed Fraud", closed_fraud, delta=f"{fraud_rate:.1f}%")

    with col5:
        avg_score = cases_df['fraud_score'].mean()
        st.metric("Avg Fraud Score", f"{avg_score:.2f}")

    # Filters
    st.markdown("---")
    st.markdown("#### Filter Cases")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        filter_status = st.multiselect(
            "Status:",
            options=cases_df['status'].unique(),
            default=['Open', 'Under Investigation']
        )

    with col2:
        filter_priority = st.multiselect(
            "Priority:",
            options=['Low', 'Medium', 'High', 'Critical'],
            default=['High', 'Critical']
        )

    with col3:
        filter_assigned = st.multiselect(
            "Assigned To:",
            options=cases_df['assigned_to'].unique(),
            default=cases_df['assigned_to'].unique()
        )

    with col4:
        filter_type = st.multiselect(
            "Case Type:",
            options=cases_df['case_type'].unique(),
            default=cases_df['case_type'].unique()
        )

    # Apply filters
    filtered_cases = cases_df[
        (cases_df['status'].isin(filter_status)) &
        (cases_df['priority'].isin(filter_priority)) &
        (cases_df['assigned_to'].isin(filter_assigned)) &
        (cases_df['case_type'].isin(filter_type))
    ]

    st.markdown(f"**Showing {len(filtered_cases)} cases**")

    # Sort options
    col1, col2 = st.columns([3, 1])

    with col1:
        sort_by = st.selectbox("Sort by:", ['created_date', 'fraud_score', 'amount', 'priority'])

    with col2:
        sort_order = st.selectbox("Order:", ['Descending', 'Ascending'])

    filtered_cases = filtered_cases.sort_values(
        by=sort_by,
        ascending=(sort_order == 'Ascending')
    )

    # Display cases table
    st.markdown("---")

    # Format dataframe for display
    display_df = filtered_cases[[
        'case_id', 'status', 'priority', 'case_type', 'customer_id',
        'amount', 'fraud_score', 'assigned_to', 'created_date'
    ]].copy()

    display_df['amount'] = display_df['amount'].apply(lambda x: f"${x:,.2f}")
    display_df['fraud_score'] = display_df['fraud_score'].apply(lambda x: f"{x:.2%}")
    display_df['created_date'] = pd.to_datetime(display_df['created_date']).dt.strftime('%Y-%m-%d %H:%M')

    # Style the dataframe
    def color_priority(val):
        if val == 'Critical':
            return 'background-color: #ff4444; color: white'
        elif val == 'High':
            return 'background-color: #ffaa44; color: white'
        elif val == 'Medium':
            return 'background-color: #ffff44'
        return ''

    styled_df = display_df.style.applymap(color_priority, subset=['priority'])

    st.dataframe(styled_df, use_container_width=True, height=400)

    # Quick actions
    st.markdown("---")
    st.markdown("#### Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        selected_case_id = st.selectbox("Select Case:", filtered_cases['case_id'].tolist())

    with col2:
        if st.button("Investigate", type="primary"):
            st.session_state.selected_case = selected_case_id
            st.success(f"Case {selected_case_id} selected for investigation. Go to 'Case Investigation' tab.")

    with col3:
        if st.button("Add Note"):
            st.info("Note dialog would open here")

    with col4:
        if st.button("Close Case"):
            st.info("Close case dialog would open here")

    # Visualizations
    st.markdown("---")
    st.markdown("#### Case Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        # Status distribution
        status_counts = filtered_cases['status'].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                    title='Cases by Status')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Priority distribution
        priority_counts = filtered_cases['priority'].value_counts()
        fig = px.bar(x=priority_counts.index, y=priority_counts.values,
                    title='Cases by Priority',
                    labels={'x': 'Priority', 'y': 'Count'},
                    color=priority_counts.index,
                    color_discrete_map={'Critical': '#ff4444', 'High': '#ffaa44',
                                       'Medium': '#ffff44', 'Low': '#44ff44'})
        st.plotly_chart(fig, use_container_width=True)


def show_case_investigation():
    st.markdown("### Case Investigation")
    st.markdown("Detailed investigation of suspicious transactions")

    if st.session_state.selected_case is None:
        st.info("👈 Please select a case from the Case Dashboard to investigate")

        # Quick case selector
        cases_df = st.session_state.cases
        open_cases = cases_df[cases_df['status'].isin(['Open', 'Under Investigation'])]

        if len(open_cases) > 0:
            st.markdown("#### Quick Select")
            case_id = st.selectbox("Select a case:", open_cases['case_id'].tolist())

            if st.button("Load Case"):
                st.session_state.selected_case = case_id
                st.rerun()

        return

    # Get case details
    case_id = st.session_state.selected_case
    case_data = st.session_state.cases[st.session_state.cases['case_id'] == case_id].iloc[0]

    # Case header
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"## 📁 {case_id}")
        st.markdown(f"**Status:** {case_data['status']} | **Priority:** {case_data['priority']}")

    with col2:
        new_status = st.selectbox("Update Status:", ['Open', 'Under Investigation', 'Escalated',
                                                      'Closed - Fraud', 'Closed - Legitimate'],
                                 index=0)

    with col3:
        if st.button("Update Status"):
            st.session_state.cases.loc[st.session_state.cases['case_id'] == case_id, 'status'] = new_status
            st.success(f"Status updated to: {new_status}")

    st.markdown("---")

    # Case overview
    st.markdown("### Case Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Fraud Score", f"{case_data['fraud_score']:.2%}")

    with col2:
        st.metric("Transaction Amount", f"${case_data['amount']:,.2f}")

    with col3:
        st.metric("Case Type", case_data['case_type'])

    with col4:
        st.metric("Assigned To", case_data['assigned_to'])

    # Transaction details
    st.markdown("---")
    st.markdown("### 💳 Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Basic Information:**")
        details_df = pd.DataFrame({
            'Field': ['Transaction ID', 'Customer ID', 'Merchant', 'Amount', 'Currency',
                     'Transaction Date', 'Card Type', 'Card Last 4'],
            'Value': [
                case_data['transaction_id'],
                case_data['customer_id'],
                case_data['merchant'],
                f"${case_data['amount']:,.2f}",
                'USD',
                case_data['created_date'].strftime('%Y-%m-%d %H:%M:%S'),
                np.random.choice(['Visa', 'Mastercard', 'Amex']),
                f"****{np.random.randint(1000, 9999)}"
            ]
        })
        st.dataframe(details_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Location & Device:**")
        location_df = pd.DataFrame({
            'Field': ['IP Address', 'Country', 'City', 'Device Type', 'Browser', 'OS'],
            'Value': [
                f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}",
                np.random.choice(['USA', 'UK', 'Nigeria', 'Russia', 'China']),
                np.random.choice(['New York', 'London', 'Lagos', 'Moscow', 'Beijing']),
                np.random.choice(['Desktop', 'Mobile', 'Tablet']),
                np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge']),
                np.random.choice(['Windows', 'MacOS', 'iOS', 'Android'])
            ]
        })
        st.dataframe(location_df, use_container_width=True, hide_index=True)

    # Triggered rules
    st.markdown("---")
    st.markdown("### Triggered Rules")

    triggered_rules = case_data['triggered_rules'].split(', ')

    for rule in triggered_rules:
        with st.expander(f"📌 {rule}", expanded=True):
            st.markdown(f"**Rule Name:** High Risk Transaction Detection")
            st.markdown(f"**Severity:** High")
            st.markdown(f"**Description:** Transaction flagged due to unusual pattern")
            st.markdown(f"**Recommendation:** Further investigation required")

    # Customer history
    st.markdown("---")
    st.markdown("### 👤 Customer History")

    # Generate sample customer history
    customer_history = pd.DataFrame({
        'Date': pd.date_range(end=datetime.now(), periods=10, freq='D')[::-1],
        'Transaction ID': [f"TXN_{np.random.randint(100000, 999999)}" for _ in range(10)],
        'Merchant': np.random.choice(['Amazon', 'eBay', 'Walmart', 'Best Buy'], 10),
        'Amount': np.random.uniform(50, 2000, 10),
        'Status': np.random.choice(['Approved', 'Approved', 'Approved', 'Flagged'], 10)
    })

    customer_history['Amount'] = customer_history['Amount'].apply(lambda x: f"${x:,.2f}")

    st.dataframe(customer_history, use_container_width=True, hide_index=True)

    # Customer statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", "347")

    with col2:
        st.metric("Avg Transaction", "$234.56")

    with col3:
        st.metric("Account Age", "2.5 years")

    with col4:
        st.metric("Previous Fraud Cases", "0")

    # Timeline visualization
    st.markdown("---")
    st.markdown("### Transaction Timeline")

    # Create sample timeline data
    timeline_data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=30, freq='H')[::-1],
        'amount': np.random.uniform(10, 500, 30)
    })

    fig = px.line(timeline_data, x='timestamp', y='amount',
                 title='Customer Transaction Pattern (Last 30 Hours)',
                 labels={'timestamp': 'Time', 'amount': 'Amount ($)'})

    # Highlight suspicious transaction
    fig.add_vline(x=case_data['created_date'], line_dash="dash", line_color="#000000",
                 annotation_text="Flagged Transaction")

    st.plotly_chart(fig, use_container_width=True)

    # Evidence and Notes
    st.markdown("---")
    st.markdown("### Investigation Notes")

    col1, col2 = st.columns([3, 1])

    with col1:
        new_note = st.text_area("Add investigation note:", height=100)

    with col2:
        note_category = st.selectbox("Category:", ["General", "Evidence", "Follow-up", "Resolution"])
        if st.button("Save Note", type="primary"):
            if new_note:
                st.success("Note saved successfully")

    # Display existing notes
    if case_data['notes_count'] > 0:
        st.markdown("**Previous Notes:**")

        for i in range(min(3, case_data['notes_count'])):
            with st.expander(f"Note {i+1} - {datetime.now() - timedelta(hours=i*2):%Y-%m-%d %H:%M}"):
                st.markdown(f"**Analyst:** {case_data['assigned_to']}")
                st.markdown(f"**Category:** {np.random.choice(['General', 'Evidence', 'Follow-up'])}")
                st.markdown("**Note:** This is a sample investigation note. Further analysis required.")

    # Actions
    st.markdown("---")
    st.markdown("### Case Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Mark as Fraud", type="primary"):
            st.success("Case marked as confirmed fraud")

    with col2:
        if st.button("Mark as Legitimate"):
            st.success("Case marked as legitimate")

    with col3:
        if st.button("Escalate Case"):
            st.warning("Case escalated to senior analyst")

    with col4:
        if st.button("Request More Info"):
            st.info("Information request sent")


def show_case_analytics():
    st.markdown("### Case Analytics")
    st.markdown("Analyze case trends and performance")

    cases_df = st.session_state.cases

    # Time range selector
    st.markdown("#### Time Range")

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input("Start Date:", value=datetime.now() - timedelta(days=30))

    with col2:
        end_date = st.date_input("End Date:", value=datetime.now())

    # Case trends
    st.markdown("---")
    st.markdown("#### Case Trends")

    # Generate daily case counts
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_cases = pd.DataFrame({
        'date': date_range,
        'new_cases': np.random.randint(0, 10, len(date_range)),
        'closed_cases': np.random.randint(0, 8, len(date_range))
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_cases['date'], y=daily_cases['new_cases'],
                            mode='lines+markers', name='New Cases'))
    fig.add_trace(go.Scatter(x=daily_cases['date'], y=daily_cases['closed_cases'],
                            mode='lines+markers', name='Closed Cases'))
    fig.update_layout(title='Daily Case Volume', xaxis_title='Date', yaxis_title='Count')

    st.plotly_chart(fig, use_container_width=True)

    # Performance metrics
    st.markdown("---")
    st.markdown("#### ⏱Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_resolution_time = np.random.uniform(2, 48)
        st.metric("Avg Resolution Time", f"{avg_resolution_time:.1f} hours")

    with col2:
        closure_rate = np.random.uniform(70, 95)
        st.metric("Case Closure Rate", f"{closure_rate:.1f}%")

    with col3:
        fraud_confirmation = np.random.uniform(60, 80)
        st.metric("Fraud Confirmation Rate", f"{fraud_confirmation:.1f}%")

    with col4:
        false_positive = np.random.uniform(10, 25)
        st.metric("False Positive Rate", f"{false_positive:.1f}%")

    # Analyst performance
    st.markdown("---")
    st.markdown("#### Analyst Performance")

    analyst_perf = pd.DataFrame({
        'Analyst': ['Analyst A', 'Analyst B', 'Analyst C'],
        'Open Cases': [12, 8, 15],
        'Closed This Month': [45, 52, 38],
        'Avg Resolution Time (hrs)': [24.5, 18.3, 32.1],
        'Accuracy': [0.92, 0.95, 0.88]
    })

    analyst_perf['Accuracy'] = analyst_perf['Accuracy'].apply(lambda x: f"{x:.1%}")

    st.dataframe(analyst_perf, use_container_width=True, hide_index=True)

    # Case distribution
    st.markdown("---")
    st.markdown("#### Case Distribution")

    col1, col2 = st.columns(2)

    with col1:
        # By case type
        case_type_counts = cases_df['case_type'].value_counts()
        fig = px.pie(values=case_type_counts.values, names=case_type_counts.index,
                    title='Cases by Type')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # By analyst
        analyst_counts = cases_df['assigned_to'].value_counts()
        fig = px.bar(x=analyst_counts.index, y=analyst_counts.values,
                    title='Cases by Analyst',
                    labels={'x': 'Analyst', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)


def show_case_workflow():
    st.markdown("### Case Workflow Configuration")
    st.markdown("Configure case management workflows and automation")

    st.markdown("#### Workflow Stages")

    workflow_stages = [
        {'Stage': 'New', 'Description': 'Case created', 'Auto-assign': True, 'SLA (hours)': 1},
        {'Stage': 'Open', 'Description': 'Assigned to analyst', 'Auto-assign': False, 'SLA (hours)': 4},
        {'Stage': 'Under Investigation', 'Description': 'Active investigation', 'Auto-assign': False, 'SLA (hours)': 24},
        {'Stage': 'Escalated', 'Description': 'Requires senior review', 'Auto-assign': True, 'SLA (hours)': 8},
        {'Stage': 'Closed', 'Description': 'Case resolved', 'Auto-assign': False, 'SLA (hours)': 0}
    ]

    workflow_df = pd.DataFrame(workflow_stages)
    st.dataframe(workflow_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Automation Rules")

    with st.expander("Auto-assignment Rules", expanded=True):
        st.markdown("**Configure automatic case assignment:**")

        col1, col2 = st.columns(2)

        with col1:
            enable_auto_assign = st.checkbox("Enable auto-assignment", value=True)
            assignment_method = st.selectbox("Assignment Method:",
                                            ["Round Robin", "Least Loaded", "Priority Based"])

        with col2:
            max_cases_per_analyst = st.number_input("Max cases per analyst:", 1, 50, 20)
            priority_weighting = st.checkbox("Weight by priority", value=True)

    with st.expander("Escalation Rules", expanded=False):
        st.markdown("**Automatic escalation triggers:**")

        col1, col2 = st.columns(2)

        with col1:
            escalate_high_amount = st.checkbox("Escalate high amounts", value=True)
            if escalate_high_amount:
                high_amount_threshold = st.number_input("Amount threshold ($):", 1000, 50000, 10000)

        with col2:
            escalate_sla = st.checkbox("Escalate on SLA breach", value=True)
            escalate_score = st.checkbox("Escalate high fraud scores", value=True)
            if escalate_score:
                score_threshold = st.slider("Fraud score threshold:", 0.0, 1.0, 0.9)

    with st.expander("Notification Rules", expanded=False):
        st.markdown("**Configure notifications:**")

        notify_new_case = st.checkbox("Notify on new case", value=True)
        notify_escalation = st.checkbox("Notify on escalation", value=True)
        notify_closure = st.checkbox("Notify on case closure", value=False)

        notification_channel = st.multiselect("Notification Channels:",
                                             ["Email", "SMS", "Slack", "In-App"],
                                             default=["Email", "In-App"])

    if st.button("Save Workflow Configuration", type="primary"):
        st.success("Workflow configuration saved successfully!")

    st.markdown("---")
    st.markdown("#### SLA Monitoring")

    sla_data = pd.DataFrame({
        'Priority': ['Critical', 'High', 'Medium', 'Low'],
        'SLA Target (hours)': [2, 8, 24, 72],
        'Avg Resolution Time': [1.8, 7.2, 20.5, 65.3],
        'SLA Compliance': ['95%', '92%', '88%', '96%'],
        'Breaches (This Month)': [2, 8, 15, 3]
    })

    st.dataframe(sla_data, use_container_width=True, hide_index=True)
