
import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# =====================================================
# Formatting Helpers
# =====================================================

def format_currency(value):
    """Format currency values for executive dashboard display."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_number(value):
    """Format large counts using K/M notation."""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def add_currency_m_column(df, value_col, label_col):
    """Create a formatted currency label column for Plotly text labels."""
    df[label_col] = df[value_col].apply(format_currency)
    return df


def format_percent(value):
    return f"{float(value):.2f}%"

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="Business Banking Client Growth & Engagement Analytics",
    layout="wide"
)

st.title("Business Banking Client Growth & Engagement Analytics")
st.caption(
    "Executive Dashboard | Client Portfolio | Campaign Performance | Branch Opportunity | Predictive Analytics"
)

# =====================================================
# Load Data
# =====================================================

conn = sqlite3.connect("business_banking_analytics.db")
df = pd.read_csv("banking_processed.csv")

branch_df = pd.read_sql_query("SELECT * FROM vw_branch_performance", conn)
segment_df = pd.read_sql_query("SELECT * FROM vw_customer_segment_kpi", conn)
monthly_df = pd.read_sql_query("SELECT * FROM vw_monthly_campaign_trend", conn)
executive_kpis = pd.read_sql_query("SELECT * FROM vw_executive_kpis", conn)
executive_highlights = pd.read_sql_query("SELECT * FROM vw_executive_highlights", conn)
campaign_province_df = pd.read_sql_query("SELECT * FROM vw_campaign_province", conn)
previous_campaign_df = pd.read_sql_query("SELECT * FROM vw_previous_campaign_outcome", conn)
contact_duration_df = pd.read_sql_query("SELECT * FROM vw_contact_duration", conn)
branch_opportunity_df = pd.read_sql_query("SELECT * FROM vw_branch_opportunity", conn)
executive_action_df = pd.read_sql_query("SELECT * FROM vw_executive_action_center", conn)

# =====================================================
# Machine Learning Model
# =====================================================

df["target"] = df["y"].map({"yes": 1, "no": 0})

features = [
    "age", "job", "marital", "education",
    "default", "housing", "loan",
    "contact", "month", "day_of_week",
    "campaign", "pdays", "previous",
    "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed",
    "province", "branch_id", "risk_score",
    "estimated_revenue", "customer_segment"
]

X = df[features]
y = df["target"]

categorical_features = X.select_dtypes(include="object").columns.tolist()
numeric_features = X.select_dtypes(exclude="object").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Feature importance
encoded_cat_features = (
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_features)
)

all_feature_names = list(encoded_cat_features) + numeric_features
rf_model = pipeline.named_steps["model"]

feature_importance = pd.DataFrame(
    {
        "feature": all_feature_names,
        "importance": rf_model.feature_importances_
    }
).sort_values("importance", ascending=False)

feature_label_map = {
    "euribor3m": "3-Month Euribor Interest Rate",
    "emp.var.rate": "Employment Variation Rate",
    "cons.price.idx": "Consumer Price Index",
    "cons.conf.idx": "Consumer Confidence Index",
    "nr.employed": "Employment Level",
    "campaign": "Number of Campaign Contacts",
    "pdays": "Days Since Previous Contact",
    "previous": "Previous Campaign Contacts",
    "estimated_revenue": "Estimated Revenue",
    "risk_score": "Risk Score",
    "age": "Age",
    "poutcome_success": "Previous Campaign Success",
    "poutcome_failure": "Previous Campaign Failure",
    "poutcome_nonexistent": "No Previous Campaign",
    "housing_yes": "Has Housing Loan",
    "housing_no": "No Housing Loan"
}

feature_importance["feature_label"] = (
    feature_importance["feature"]
    .replace(feature_label_map)
    .str.replace("_", " ", regex=False)
    .str.title()
)

# Priority customers
results = X_test.copy()
results["customer_id"] = df.loc[X_test.index, "customer_id"]
results["subscription_probability"] = y_proba
results["predicted_response"] = y_pred
results["actual_response"] = y_test.values

results = results.sort_values("subscription_probability", ascending=False)

# =====================================================
# Executive KPI Cards
# =====================================================

kpi = executive_kpis.iloc[0]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Business Clients", format_number(kpi["total_business_clients"]))

with col2:
    st.metric("Total Revenue", format_currency(kpi["total_revenue"]))

with col3:
    st.metric("Campaign Conversion", f"{kpi['campaign_conversion_pct']:.2f}%")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Portfolio Growth", f"{kpi['portfolio_growth_pct']:.1f}%")

with col5:
    st.metric("High Value Clients", format_number(kpi["high_value_clients"]))

with col6:
    st.metric("Digital Adoption", f"{kpi['digital_adoption_pct']:.2f}%")

# =====================================================
# Executive Insights
# =====================================================

highlight = executive_highlights.iloc[0]

st.subheader("Executive Insights")
st.caption("Automatically generated from client segment, branch, and campaign performance data.")

h1, h2 = st.columns(2)

with h1:
    st.success(f"""
**Highest Performing Client Segment**

### {highlight['highest_performing_segment']}

Avg Revenue: **{format_currency(highlight["highest_segment_revenue"])}**
Subscription Rate: **{highlight['highest_segment_conversion']:.2f}%**

**Recommendation:** Prioritize this segment for premium product cross-sell and relationship manager outreach.
""")

with h2:
    st.success(f"""
**Highest Growth Opportunity Branch**

### {highlight['highest_opportunity_region']}

Opportunity Score: **{highlight['highest_opportunity_score']:.2f}**

**Recommendation:** Prioritize this region for targeted growth actions based on revenue, client base, and untapped conversion potential.
""")

h3, h4 = st.columns(2)

with h3:
    st.success(f"""
**Highest Converting Campaign Period**

### {highlight['best_campaign']}

Conversion Rate: **{highlight['best_campaign_conversion']:.2f}%**

**Recommendation:** Review campaign timing, targeting, and contact strategy from this period before the next campaign cycle.
""")

with h4:
    st.warning(f"""
**Lowest Conversion Segment**

### {highlight['lowest_engagement_segment']}

Subscription Rate: **{highlight['lowest_segment_conversion']:.2f}%**

**Recommendation:** Improve messaging, offer design, or channel strategy before prioritizing this segment.
""")

st.divider()

# =====================================================
# Client Portfolio
# =====================================================

st.header("Client Portfolio")
st.caption("Which client segments generate the greatest business value?")

segment_df = add_currency_m_column(segment_df, "total_revenue", "total_revenue_label")
segment_df = add_currency_m_column(segment_df, "revenue_per_client", "revenue_per_client_label")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Revenue by Client Segment")

    fig = px.bar(
        segment_df,
        x="customer_segment",
        y="total_revenue",
        color="customer_segment",
        text="total_revenue_label"
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside")

    fig.update_layout(
        xaxis_title="Client Segment",
        yaxis_title="Total Revenue ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2s",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Client Portfolio Composition")

    fig = px.pie(
        segment_df,
        names="customer_segment",
        values="total_customers",
        hole=0.45
    )

    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Clients: %{value:,.0f}<br>Share: %{percent}<extra></extra>"
    )

    fig.update_layout(height=380, margin=dict(t=20, b=40, l=40, r=20))

    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Revenue per Client by Segment")

    fig = px.bar(
        segment_df,
        x="customer_segment",
        y="revenue_per_client",
        color="customer_segment",
        text="revenue_per_client_label"
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside")

    fig.update_layout(
        xaxis_title="Client Segment",
        yaxis_title="Revenue per Client ($)",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Cross-Sell Opportunity by Segment")

    fig = px.bar(
        segment_df.sort_values("cross_sell_opportunity_score"),
        x="cross_sell_opportunity_score",
        y="customer_segment",
        orientation="h",
        text="cross_sell_opportunity_score"
    )

    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")

    fig.update_layout(
        xaxis_title="Opportunity Score",
        yaxis_title="Client Segment",
        height=380,
        margin=dict(t=20, b=40, l=40, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================
# Campaign Performance
# =====================================================

st.header("Client Acquisition & Campaign Performance")
st.caption("Which client groups, regions, and contact strategies drive campaign conversion? Campaign activity is uneven across months, so monthly peaks should be interpreted as campaign timing effects.")

month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_labels = {
    "mar": "Mar", "apr": "Apr", "may": "May", "jun": "Jun", "jul": "Jul",
    "aug": "Aug", "sep": "Sep", "oct": "Oct", "nov": "Nov", "dec": "Dec"
}
monthly_df["month_label"] = monthly_df["month"].map(month_labels)
monthly_df["month"] = pd.Categorical(monthly_df["month"], categories=month_order, ordered=True)
monthly_df = monthly_df.sort_values("month")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Campaign Conversion Trend")

    fig = px.line(
        monthly_df,
        x="month_label",
        y="conversion_rate_pct",
        markers=True
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Conversion Rate (%)",
        height=380,
        margin=dict(t=20, b=40, l=40, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Conversion Rate by Client Segment")

    fig = px.bar(
        segment_df,
        x="customer_segment",
        y="subscription_rate_pct",
        color="customer_segment",
        text="subscription_rate_pct"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig.update_layout(
        xaxis_title="Client Segment",
        yaxis_title="Conversion Rate (%)",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Campaign Success by Province")

    fig = px.bar(
        campaign_province_df.sort_values("conversion_rate_pct"),
        x="conversion_rate_pct",
        y="province",
        orientation="h",
        text="conversion_rate_pct"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig.update_layout(
        xaxis_title="Conversion Rate (%)",
        yaxis_title="Province",
        height=380,
        margin=dict(t=20, b=40, l=40, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Conversion by Previous Campaign Outcome")

    previous_campaign_df["previous_campaign_outcome"] = (
        previous_campaign_df["previous_campaign_outcome"]
        .replace({
            "success": "Previous Success",
            "failure": "Previous Failure",
            "nonexistent": "No Previous Campaign"
        })
    )

    fig = px.bar(
        previous_campaign_df,
        x="previous_campaign_outcome",
        y="conversion_rate_pct",
        color="previous_campaign_outcome",
        text="conversion_rate_pct"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig.update_layout(
        xaxis_title="Previous Campaign Outcome",
        yaxis_title="Current Conversion Rate (%)",
        showlegend=False,
        height=380,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Call Duration by Conversion Outcome")

contact_duration_df["term_deposit_subscribed"] = (
    contact_duration_df["term_deposit_subscribed"]
    .replace({"yes": "Subscribed", "no": "Not Subscribed"})
)

fig = px.box(
    contact_duration_df,
    x="term_deposit_subscribed",
    y="duration",
    color="term_deposit_subscribed",
    points=False
)

fig.update_layout(
    xaxis_title="Campaign Outcome",
    yaxis_title="Call Duration",
    showlegend=False,
    height=420,
    margin=dict(t=20, b=40, l=40, r=20)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================
# Branch Performance
# =====================================================

st.header("Branch & Regional Performance")
st.caption("Which branches generate value, convert clients, and offer the highest growth opportunity?")

branch_opportunity_df = add_currency_m_column(branch_opportunity_df, "total_revenue", "total_revenue_label")
branch_opportunity_df = add_currency_m_column(branch_opportunity_df, "revenue_per_client", "revenue_per_client_label")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Total Revenue by Branch")

    fig = px.bar(
        branch_opportunity_df,
        x="branch_name",
        y="total_revenue",
        color="province",
        text="total_revenue_label"
    )

    fig.update_traces(texttemplate="%{text}", textposition="outside")

    fig.update_layout(
        xaxis_title="Branch",
        yaxis_title="Total Revenue ($)",
        yaxis_tickprefix="$",
        yaxis_tickformat=".2s",
        height=380,
        margin=dict(t=20, b=70, l=40, r=20)
    )

    fig.update_xaxes(tickangle=20)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Campaign Conversion Rate by Branch")

    fig = px.bar(
        branch_opportunity_df,
        x="branch_name",
        y="conversion_rate_pct",
        color="province",
        text="conversion_rate_pct"
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig.update_layout(
        xaxis_title="Branch",
        yaxis_title="Conversion Rate (%)",
        height=380,
        margin=dict(t=20, b=70, l=40, r=20)
    )

    fig.update_xaxes(tickangle=20)

    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Branch Growth Opportunity Ranking")

    fig = px.bar(
        branch_opportunity_df.sort_values("branch_opportunity_score"),
        x="branch_opportunity_score",
        y="branch_name",
        orientation="h",
        text="branch_opportunity_score"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    fig.update_layout(
        xaxis_title="Opportunity Score",
        yaxis_title="Branch",
        height=380,
        margin=dict(t=20, b=40, l=40, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.subheader("Branch Opportunity Drivers")

    component_df = branch_opportunity_df[
        ["branch_name", "revenue_score", "client_base_score", "untapped_conversion_score"]
    ].melt(
        id_vars="branch_name",
        var_name="component",
        value_name="score"
    )

    component_df["component"] = component_df["component"].replace({
        "revenue_score": "Revenue Score",
        "client_base_score": "Client Base Score",
        "untapped_conversion_score": "Untapped Conversion Score"
    })

    fig = px.bar(
        component_df,
        x="branch_name",
        y="score",
        color="component",
        barmode="group"
    )

    fig.update_layout(
        xaxis_title="Branch",
        yaxis_title="Component Score",
        legend_title="",
        height=380,
        margin=dict(t=20, b=70, l=40, r=20)
    )

    fig.update_xaxes(tickangle=20)

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================
# Executive Action Center
# =====================================================

st.header("Executive Action Center")
st.caption("Growth opportunities and recommended actions by client segment and region.")

st.subheader("Top Growth Opportunities by Recommendation Area")

display_cols = [
    "recommendation_area",
    "business_rule",
    "rank",
    "opportunity",
    "opportunity_score",
    "opportunity_potential",
    "recommended_action"
]

action_display_df = executive_action_df[display_cols].rename(
    columns={
        "recommendation_area": "Recommendation Area",
        "business_rule": "Business Rule",
        "rank": "Rank",
        "opportunity": "Opportunity",
        "opportunity_score": "Opportunity Score",
        "opportunity_potential": "Opportunity Potential",
        "recommended_action": "Recommended Action"
    }
)

st.dataframe(
    action_display_df,
    use_container_width=True,
    hide_index=True
)

st.subheader("Recommended Executive Actions")

areas = executive_action_df["recommendation_area"].unique()
col1, col2 = st.columns(2)

for i, area in enumerate(areas):
    area_df = executive_action_df[
        executive_action_df["recommendation_area"] == area
    ].sort_values("rank")

    top = area_df.iloc[0]

    card = f"""
**Business Rule**

{top["business_rule"]}

---

**Top Opportunities**

1. {area_df.iloc[0]["opportunity"]} *(Score: {area_df.iloc[0]["opportunity_score"]:.1f})*
2. {area_df.iloc[1]["opportunity"]} *(Score: {area_df.iloc[1]["opportunity_score"]:.1f})*
3. {area_df.iloc[2]["opportunity"]} *(Score: {area_df.iloc[2]["opportunity_score"]:.1f})*

---

**Recommended Action**

{top["recommended_action"]}
"""

    if i % 2 == 0:
        with col1:
            with st.container(border=True):
                st.markdown(f"### {area}")
                st.markdown(card)
    else:
        with col2:
            with st.container(border=True):
                st.markdown(f"### {area}")
                st.markdown(card)

st.divider()

# =====================================================
# Predictive Analytics
# =====================================================

st.header("Predictive Analytics")
st.caption("Customer prioritization model for campaign conversion and outreach planning.")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{accuracy:.2%}")

with col2:
    st.metric("Precision", f"{precision:.2%}")

with col3:
    st.metric("Recall", f"{recall:.2%}")

with col4:
    st.metric("ROC AUC", f"{roc_auc:.3f}")

st.caption("Model prioritizes likely subscribers to support targeted outreach while reducing unnecessary campaign contact.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"]
    )

    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues"
    )

    fig.update_layout(height=380, margin=dict(t=20, b=40, l=40, r=20))

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("ROC Curve")

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    roc_df = pd.DataFrame(
        {
            "False Positive Rate": fpr,
            "True Positive Rate": tpr
        }
    )

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(dash="dash")
    )

    fig.update_layout(height=380, margin=dict(t=20, b=40, l=40, r=20))

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Key Drivers of Campaign Subscription")

top_features = feature_importance.head(10).sort_values("importance")

fig = px.bar(
    top_features,
    x="importance",
    y="feature_label",
    orientation="h",
    text="importance"
)

fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

fig.update_layout(
    xaxis_title="Relative Importance",
    yaxis_title="",
    height=430,
    margin=dict(t=20, b=40, l=40, r=40)
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Top 20 Priority Customers")

priority_results = results.copy()

def priority_level(prob):
    if prob >= 0.90:
        return "Very High"
    elif prob >= 0.80:
        return "High"
    elif prob >= 0.70:
        return "Medium"
    else:
        return "Low"

def recommended_action(prob):
    if prob >= 0.90:
        return "Immediate Relationship Manager Follow-up"
    elif prob >= 0.80:
        return "Priority Campaign"
    elif prob >= 0.70:
        return "Standard Campaign"
    else:
        return "Monitor"

priority_results["priority_level"] = priority_results["subscription_probability"].apply(priority_level)
priority_results["recommended_action"] = priority_results["subscription_probability"].apply(recommended_action)
priority_results["subscription_probability"] = (priority_results["subscription_probability"] * 100).round(1)
priority_results["estimated_revenue"] = priority_results["estimated_revenue"].apply(format_currency)
priority_results["subscription_probability"] = priority_results["subscription_probability"].astype(str) + "%"

display_cols = [
    "customer_id",
    "province",
    "branch_id",
    "customer_segment",
    "estimated_revenue",
    "subscription_probability",
    "priority_level",
    "recommended_action"
]

priority_results = priority_results[display_cols].rename(
    columns={
        "customer_id": "Customer ID",
        "province": "Province",
        "branch_id": "Branch ID",
        "customer_segment": "Customer Segment",
        "estimated_revenue": "Estimated Revenue",
        "subscription_probability": "Subscription Probability (%)",
        "priority_level": "Priority Level",
        "recommended_action": "Recommended Action"
    }
)

st.dataframe(
    priority_results.head(20),
    use_container_width=True,
    hide_index=True
)

st.info(
    "Model insight: customers with stronger client value indicators, previous campaign success, "
    "and favorable economic or campaign context receive higher predicted conversion probabilities."
)