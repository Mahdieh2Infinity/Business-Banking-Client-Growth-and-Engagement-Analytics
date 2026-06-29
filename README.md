# Business Banking Client Growth & Engagement Analytics

## Project Overview

This project is an end-to-end business banking analytics solution built with Python, SQL, Machine Learning, and Streamlit.

It transforms raw banking campaign data into executive-ready insights through preprocessing, feature engineering, relational database design, reusable SQL KPI views, interactive dashboards, and predictive analytics.

The project simulates how a Canadian business banking team could use customer, campaign, branch, and economic data to identify growth opportunities, improve client engagement, and prioritize campaign outreach.

---

## Business Problem

Business banking teams need to understand which clients, regions, and campaign strategies offer the greatest opportunity for growth.

The objective of this project is to:

- Identify high-value client segments
- Analyze branch and regional performance
- Measure campaign conversion trends
- Surface cross-sell and growth opportunities
- Predict which clients are most likely to subscribe to a banking product
- Recommend executive actions by segment and region

---

## Solution Architecture

### Step 1 — Data Preparation & Feature Engineering

- Cleaned missing values
- Created customer IDs
- Added synthetic Canadian provinces and branch IDs
- Created risk score
- Created estimated revenue using pre-campaign customer attributes
- Created customer segments based on estimated revenue

### Step 2 — Relational SQL Database Design

Created normalized tables for:

- Customers
- Branches
- Financial Profiles
- Campaign Contacts
- Campaign Outcomes
- Economic Indicators

### Step 3 — SQLite Database Creation

Loaded normalized tables into a SQLite database.

### Step 4 — SQL Analytics & KPI Layer

Developed reusable SQL queries and views for:

- Branch performance
- Customer segment KPIs
- Monthly campaign trends
- Executive KPI cards
- Executive highlights
- Campaign performance by province
- Previous campaign outcome analysis
- Branch opportunity scoring
- Executive action recommendations

### Step 5 — Predictive Analytics Layer

Built a Random Forest classification model to predict customer subscription likelihood.

The model uses customer demographics, financial profile indicators, campaign history, branch information, customer value indicators, and economic variables.

To avoid target leakage, call duration was excluded from the predictive feature set.

### Step 6 — Streamlit Dashboard

Built an interactive Streamlit dashboard with:

- Executive KPI cards
- Executive insights
- Client portfolio analytics
- Campaign performance analytics
- Branch and regional performance
- Executive action center
- Predictive analytics model results
- Priority customer recommendations

---

## Technology Stack

- Python
- Pandas
- NumPy
- SQLite
- SQL
- Scikit-learn
- Plotly
- Streamlit
- Matplotlib

---

## Dashboard Features

### Executive Dashboard

- Total business clients
- Total estimated revenue
- Campaign conversion rate
- Portfolio growth
- High-value clients
- Digital adoption

### Client Portfolio

- Total revenue by client segment
- Client portfolio composition
- Revenue per client
- Cross-sell opportunity score

### Campaign Performance

- Monthly campaign conversion trend
- Conversion rate by client segment
- Campaign success by province
- Conversion by previous campaign outcome
- Call duration by conversion outcome

### Branch & Regional Performance

- Total revenue by branch
- Campaign conversion rate by branch
- Branch growth opportunity ranking
- Branch opportunity drivers

### Executive Action Center

- Growth opportunities by recommendation area
- Business rules
- Opportunity scores
- Recommended executive actions

### Predictive Analytics

- Accuracy
- Precision
- Recall
- ROC AUC
- Confusion matrix
- ROC curve
- Key drivers of campaign subscription
- Top priority customers for outreach

---

## Machine Learning Approach

The predictive model estimates the likelihood of customer subscription using a Random Forest classifier.

Key modeling decisions:

- Excluded call duration to avoid target leakage
- Included estimated revenue because it was engineered from pre-campaign attributes
- Included customer segment as a business value indicator
- Used class weighting to handle imbalance between subscribers and non-subscribers

Key predictive drivers include:

- 3-Month Euribor Interest Rate
- Estimated Revenue
- Age
- Employment Indicators
- Campaign Contact History
- Consumer Confidence
- Previous Campaign Outcome

---

## Business Value

This project demonstrates how analytics can support business banking decision-making by helping teams:

- Prioritize high-value client segments
- Identify under-converted growth opportunities
- Improve campaign targeting
- Support relationship manager outreach
- Monitor branch-level performance
- Convert predictive model outputs into actionable recommendations

---

## Future Enhancements

- Hyperparameter tuning
- Model calibration
- SHAP-based explainability
- Customer lifetime value modeling
- Multi-product recommendation engine
- Cloud database integration
- Automated data refresh
- Role-based dashboard views

---

## Author

Mahdieh Ghafourian (Aylin)
Toronto, Ontario
Master of Information Systems & Technology, York University
