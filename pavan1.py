import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.neighbors import NearestNeighbors

# Load datasets
customers_df = pd.read_csv(r"C:\Users\nagal\Downloads\archive (1)\olist_customers_dataset.csv")
orders_df = pd.read_csv(r"C:\Users\nagal\Downloads\archive (1)\olist_orders_dataset.csv")
order_payments_df = pd.read_csv(r"C:\Users\nagal\Downloads\archive (1)\olist_order_payments_dataset.csv")
order_items_df = pd.read_csv(r"C:\Users\nagal\Downloads\archive (1)\olist_order_items_dataset.csv")

# Convert timestamps to datetime format
orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"], errors="coerce")

# Merge datasets
customer_orders_df = pd.merge(customers_df, orders_df, on="customer_id", how="inner")
orders_payments_df = pd.merge(orders_df, order_payments_df, on="order_id", how="inner")
orders_items_df = pd.merge(customer_orders_df, order_items_df, on="order_id", how="inner")  # Fixes 'NameError'

# Fraud Detection using Isolation Forest (Real-Time Anomaly Monitoring)
fraud_detector = IsolationForest(contamination=0.01, random_state=42)
order_payments_df["fraud_score"] = fraud_detector.fit_predict(order_payments_df[["payment_value"]])
order_payments_df["fraud_alert"] = np.where(order_payments_df["fraud_score"] == -1, "Suspicious", "Normal")

# Real-Time Fraud Alerts
st.sidebar.title("🚨 Fraud Alert System")
if "Suspicious" in order_payments_df["fraud_alert"].values:
    st.sidebar.warning("⚠ Suspicious transactions detected! Please review fraud alerts.")

# Customer Segmentation
customer_segmentation = customer_orders_df.groupby("customer_unique_id")["order_id"].count().reset_index()
customer_segmentation.columns = ["customer_unique_id", "total_orders"]
customer_segmentation["customer_type"] = np.where(customer_segmentation["total_orders"] > 1, "Returning Customer", "New Customer")

# AI-Powered Customer Churn Prediction
customer_segmentation["churn_risk"] = np.where(customer_segmentation["total_orders"] <= 2, 1, 0)
X_churn = customer_segmentation[["total_orders"]]
y_churn = customer_segmentation["churn_risk"]
X_train, X_test, y_train, y_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
churn_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
churn_model.fit(X_train, y_train)
y_pred_churn = churn_model.predict(X_test)
churn_accuracy = accuracy_score(y_test, y_pred_churn)

# Customer Trend Analysis (Fixed Seasonality Issue)
if "order_purchase_timestamp" in orders_df.columns:
    monthly_orders = orders_df.set_index("order_purchase_timestamp").resample("M")["order_id"].count()

    # Ensure no zero/negative values before applying multiplicative model
    if (monthly_orders > 0).all():
        decomposition = seasonal_decompose(monthly_orders, model="multiplicative")
    else:
        decomposition = seasonal_decompose(monthly_orders, model="additive")

    trend_data = decomposition.trend.dropna()
    seasonal_data = decomposition.seasonal.dropna()
else:
    trend_data = None
    seasonal_data = None

# AI-Powered Product Recommendations (Memory-Optimized)
recommended_products = ["No recommendations available - insufficient data"]
if not orders_items_df.empty and {"customer_unique_id", "product_id"}.issubset(orders_items_df.columns):
    # Get top 1000 most popular products to limit matrix size
    top_products = orders_items_df["product_id"].value_counts().nlargest(1000).index.tolist()
    filtered_orders = orders_items_df[orders_items_df["product_id"].isin(top_products)]

    if not filtered_orders.empty:
        # Get active customers (those who bought at least 2 products)
        active_customers = filtered_orders.groupby("customer_unique_id")["product_id"].nunique()
        active_customers = active_customers[active_customers >= 2].index.tolist()

        if len(active_customers) >= 5:  # Need at least 5 active customers
            # Create sparse matrix using scipy for memory efficiency
            try:
                from scipy.sparse import csr_matrix

                # Create mapping dictionaries
                customer_map = {c: i for i, c in enumerate(active_customers)}
                product_map = {p: i for i, p in enumerate(top_products)}

                # Prepare data for sparse matrix
                rows = []
                cols = []
                data = []
                for _, row in filtered_orders.iterrows():
                    if row["customer_unique_id"] in customer_map and row["product_id"] in product_map:
                        rows.append(customer_map[row["customer_unique_id"]])
                        cols.append(product_map[row["product_id"]])
                        data.append(1)  # Binary indicator (1 = purchased)
                # Create sparse matrix
                sparse_matrix = csr_matrix((data, (rows, cols)),
                                        shape=(len(active_customers), len(top_products)))

                # Use approximate nearest neighbors for large datasets
                model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
                model.fit(sparse_matrix)

                # Get random customer and find similar ones
                random_idx = np.random.randint(0, len(active_customers))
                distances, indices = model.kneighbors(sparse_matrix[random_idx])

                # Get recommended products
                similar_customers = [active_customers[i] for i in indices[0]]
                recommended_products = filtered_orders[filtered_orders["customer_unique_id"].isin(similar_customers)] \
                    .groupby("product_id").size().nlargest(5).index.tolist()

            except Exception as e:
                st.warning(f"Could not generate recommendations: {str(e)}")
                recommended_products = ["No recommendations available - error in processing"]

# AI-Powered Sales Forecasting with LSTM
if "order_purchase_timestamp" in orders_df.columns:
    X_train = np.array(monthly_orders.values[:-6]).reshape(-1, 1, 1)
    y_train = np.array(monthly_orders.values[1:-5]).reshape(-1, 1)

    lstm_model = Sequential([
        LSTM(50, activation="relu", input_shape=(1,1), return_sequences=True),
        LSTM(50, activation="relu"),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(X_train, y_train, epochs=50, verbose=0)

    X_forecast = np.array(monthly_orders.values[-6:]).reshape(-1, 1, 1)
    forecast = lstm_model.predict(X_forecast).flatten()
else:
    forecast = ["Timestamp data missing for forecasting"]

# Streamlit Dashboard
st.title("📊 Customer 360 AI Analytics Dashboard")
st.markdown("### Real-Time Customer Insights & Predictive Analytics")

# Dashboard Overview
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", customer_segmentation["customer_unique_id"].nunique())
with col2:
    st.metric("Active Customers", len(active_customers) if 'active_customers' in locals() else "N/A")
with col3:
    st.metric("Fraud Alerts", order_payments_df["fraud_alert"].value_counts().get("Suspicious", 0))

st.markdown("---")

# 📍 Customer Trend Analysis
st.subheader("📈 Customer Purchase Trends")
if trend_data is not None and seasonal_data is not None:
    fig, ax = plt.subplots()
    ax.plot(trend_data.index, trend_data.values, marker="o", linestyle="-", color="blue", label="Trend")
    ax.plot(seasonal_data.index, seasonal_data.values, linestyle="dashed", color="red", label="Seasonality")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("Trend analysis unavailable due to missing timestamp data.")

# 📍 Customer Segmentation Pie Chart
st.subheader("🎯 Customer Segmentation Breakdown")
fig = px.pie(customer_segmentation, names="customer_type", values="total_orders", title="Customer Segmentation")
st.plotly_chart(fig)

# 📍 Customer Churn Prediction
st.subheader("🔮 Customer Churn Risk Prediction")
st.write(f"Model Accuracy: {churn_accuracy * 100:.2f}%")

# 📍 Fraud Detection Table (Real-Time Monitoring)
st.subheader("🔍 Fraud Detection: Suspicious Transactions")
st.dataframe(order_payments_df[order_payments_df["fraud_alert"] == "Suspicious"])

# 📍 AI-Powered Sales Forecasting
st.subheader("📈 Sales Forecast (Next 6 Months)")
fig, ax = plt.subplots()
ax.plot(monthly_orders.index, monthly_orders.values, marker="o", linestyle="-", color="blue", label="Historical Sales")
ax.plot(pd.date_range(monthly_orders.index[-1], periods=6, freq="M"), forecast, linestyle="dashed", color="red", label="Forecast")
ax.legend()
st.pyplot(fig)

# 📍 Customer Lifetime Value (CLV) Prediction
st.subheader("💰 Customer Lifetime Value Prediction")
if not orders_items_df.empty:
    # Calculate CLV
    clv_df = orders_items_df.groupby('customer_unique_id')['price'].sum().reset_index()
    clv_df.columns = ['Customer', 'Lifetime Value']
    st.dataframe(clv_df.sort_values('Lifetime Value', ascending=False).head(10))
else:
    st.warning("Insufficient data for CLV calculation")

# 📍 Dynamic Fraud Heatmaps
st.subheader("🔥 Fraud Risk Heatmap")
if not order_payments_df.empty:
    # Merge timestamp from orders_df to order_payments_df
    fraud_data = pd.merge(order_payments_df, orders_df[['order_id', 'order_purchase_timestamp']], on='order_id')
    fraud_heatmap = fraud_data.groupby(
        [fraud_data['order_purchase_timestamp'].dt.date, 'fraud_alert']).size().unstack().fillna(0)

    fig, ax = plt.subplots()
    sns.heatmap(fraud_heatmap.T, cmap='YlOrRd', ax=ax)
    ax.set_title('Fraud Alerts Over Time')
    st.pyplot(fig)
else:
    st.warning("Insufficient data for fraud heatmap")

# 📍 AI-Driven Personalized Discounts
st.subheader("🎯 Personalized Retention Offers")
if 'churn_model' in locals() and not customer_segmentation.empty:
    high_risk_customers = customer_segmentation[customer_segmentation['churn_risk'] == 1]
    if not high_risk_customers.empty:
        st.write("Suggested discounts for high-risk churn customers:")
        discount_strategy = high_risk_customers[['customer_unique_id', 'total_orders']]
        discount_strategy['discount'] = np.where(
            discount_strategy['total_orders'] > 2, '15%', '10%')
        st.dataframe(discount_strategy.head(10))
    else:
        st.info("No high-risk churn customers identified")
else:
    st.warning("Churn model not available for discount recommendations")

# 📍 Real-Time Product Demand Forecasting
st.subheader("📈 Next Month's Product Trends")
if not orders_items_df.empty and 'order_purchase_timestamp' in orders_items_df.columns:
    # Simple trend analysis for demonstration
    product_trends = orders_items_df.groupby(
        ['product_id', orders_items_df['order_purchase_timestamp'].dt.to_period('M')])['price'].sum().unstack()
    forecast = product_trends.iloc[:, -1].nlargest(5)
    st.write("Top 5 predicted trending products:")
    st.write(forecast)
else:
    st.warning("Insufficient data for product demand forecasting")

# 📍 AI-Powered Customer Satisfaction Predictions
st.subheader("😊 Customer Satisfaction Predictions")
if not orders_items_df.empty:
    satisfaction_df = orders_items_df.groupby('customer_unique_id').agg(
        total_orders=('order_id', 'nunique'),
        total_spend=('price', 'sum')
    ).reset_index()
    satisfaction_df['satisfaction_score'] = np.where(
        (satisfaction_df['total_orders'] > 2) & (satisfaction_df['total_spend'] > 100), 'High', 'Medium')
    st.dataframe(satisfaction_df.sort_values('total_spend', ascending=False).head(10))
else:
    st.warning("Insufficient data for satisfaction predictions")

# 📍 Fraud Alert Notifications
st.subheader("🔔 Fraud Alert Notifications")
if "Suspicious" in order_payments_df["fraud_alert"].values:
    suspicious_count = order_payments_df["fraud_alert"].value_counts()["Suspicious"]
    st.error(f"🚨 {suspicious_count} suspicious transactions detected!")
    if st.button("View Fraud Details", key="fraud_details_1"):
        st.dataframe(order_payments_df[order_payments_df["fraud_alert"] == "Suspicious"])
else:
    st.success("No fraud alerts currently")

# 📍 Customizable Customer Segmentation
st.subheader("🔍 Custom Customer Segmentation")
segmentation_options = st.multiselect(
    "Select segmentation criteria:",
    options=['total_orders', 'customer_type', 'churn_risk'],
    default=['customer_type']
)
if segmentation_options:
    segment_view = customer_segmentation.groupby(segmentation_options).size().reset_index(name='count')
    st.dataframe(segment_view)

# 📍 AI-Powered Product Bundle Recommendations
st.subheader("🎁 Recommended Product Bundles")
if len(recommended_products) >= 2:
    bundles = []
    for i in range(min(3, len(recommended_products)-1)):
        bundles.append(f"{recommended_products[i]} + {recommended_products[i+1]}")
    st.write("Frequently purchased together:")
    for bundle in bundles:
        st.write(f"- {bundle}")
else:
    st.warning("Insufficient data for bundle recommendations")

# 📍 AI-Powered Customer Sentiment Analysis
st.subheader("😊 Customer Sentiment Analysis")
if not orders_items_df.empty:
    sentiment_df = orders_items_df.groupby('customer_unique_id').agg(
        total_orders=('order_id', 'nunique'),
        total_spend=('price', 'sum')
    ).reset_index()
    sentiment_df['sentiment'] = np.where(
        (sentiment_df['total_orders'] > 2) & (sentiment_df['total_spend'] > 100), 'Positive', 'Neutral')
    st.dataframe(sentiment_df.sort_values('total_spend', ascending=False).head(10))
else:
    st.warning("Insufficient data for sentiment analysis")

# 📍 Dynamic Customer Segmentation Dashboard
st.subheader("📊 Interactive Customer Segmentation")
col1, col2 = st.columns(2)
with col1:
    segment_by = st.selectbox(
        "Segment by:", options=['customer_type', 'churn_risk', 'total_orders'], index=0)
with col2:
    metric = st.selectbox(
        "View metric:", options=['count', 'total_spend'], index=0)

if segment_by and metric:
    if metric == 'total_spend':
        segment_data = orders_items_df.groupby(['customer_unique_id', segment_by])['price'].sum().reset_index()
    else:
        segment_data = customer_segmentation

fig = px.bar(segment_data.groupby(segment_by).size().reset_index(name='count'),
            x=segment_by, y='count', title=f"Customer Segmentation by {segment_by}")
st.plotly_chart(fig)

# 📍 AI-Driven Inventory Optimization
st.subheader("📦 Inventory Optimization Forecast")
if not orders_items_df.empty and 'order_purchase_timestamp' in orders_items_df.columns:
    inventory_forecast = orders_items_df.groupby(
        ['product_id', orders_items_df['order_purchase_timestamp'].dt.to_period('M')])['price'].sum().unstack()
    forecast = inventory_forecast.iloc[:, -1].nlargest(5)
    st.write("Top 5 products to stock up:")
    st.write(forecast)
else:
    st.warning("Insufficient data for inventory forecasting")

# 📍 Personalized AI-Powered Recommendations
st.subheader("🛍 AI-Powered Personalized Recommendations")
st.write(recommended_products)
