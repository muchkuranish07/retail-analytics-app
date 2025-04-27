# --- IMPORTS ---
import streamlit as st
import psycopg2
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(page_title="Retail Web App", page_icon="🛒")

# --- DATABASE CONNECTION ---
def create_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="retail_data",
        user="postgres",
        password="Retail123!"
    )

# --- LOGIN PAGE ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🔐 Login Page")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        email = st.text_input("Email Address")
        submit = st.form_submit_button("Login")

    if submit:
        if username and password and email:
            st.session_state['logged_in'] = True
            st.success(f"Welcome, {username}!")
        else:
            st.error("Please fill all fields!")

# --- MAIN APPLICATION ---
if st.session_state['logged_in']:
    st.title("🏠 Household Purchase Lookup")

    hshd_num = st.text_input("Enter Household Number (HSHD_NUM):")
    if hshd_num:
        try:
            conn = create_connection()
            query = """
            SELECT t.basket_num, t.purchase_date, t.product_num, p.department, p.commodity, t.spend, t.units
            FROM transactions t
            JOIN products p ON t.product_num = p.product_num
            WHERE t.hshd_num = %s
            ORDER BY t.purchase_date;
            """
            df = pd.read_sql(query, conn, params=(hshd_num,))
            conn.close()

            if df.empty:
                st.warning(f"No transactions found for Household {hshd_num}.")
            else:
                st.success(f"Found {len(df)} transactions!")
                st.dataframe(df)

                st.subheader("📊 Summary Statistics")
                st.write(f"Total Spend: ${df['spend'].sum():.2f}")
                st.write(f"Total Items Purchased: {df['units'].sum()} units")

                st.subheader("🛒 Spend by Department")
                st.bar_chart(df.groupby('department')['spend'].sum())

        except Exception as e:
            st.error(f"Error fetching data: {e}")

    # --- MACHINE LEARNING SECTION ---

    st.header("🧠 Basket Analysis")

    if st.button("Run Basket Analysis", key="basket_button"):
        try:
            conn = create_connection()
            df_basket = pd.read_sql("SELECT basket_num, product_num FROM transactions LIMIT 5000", conn)
            conn.close()

            basket = df_basket.pivot_table(index='basket_num', columns='product_num', aggfunc=lambda x: 1, fill_value=0)

            if basket.shape[1] > 1:
                target = basket.columns[0]
                X = basket.drop(columns=[target])
                y = basket[target]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                model = RandomForestClassifier()
                model.fit(X_train, y_train)

                accuracy = model.score(X_test, y_test)
                st.success(f"Basket Analysis Model Accuracy: {accuracy*100:.2f}%")

                importances = pd.Series(model.feature_importances_, index=X.columns)
                st.subheader("Top Products Associated")
                st.bar_chart(importances.sort_values(ascending=False).head(5))
            else:
                st.warning("Not enough product columns to train model.")
        except Exception as e:
            st.error(f"Basket Analysis Error: {e}")

    # --- CHURN PREDICTION SECTION ---
    st.header("📉 Churn Prediction")

    if st.button("Run Churn Prediction", key="churn_button"):
        try:
            conn = create_connection()
            df_households = pd.read_sql("SELECT * FROM households", conn)
            df_transactions = pd.read_sql("SELECT hshd_num, spend FROM transactions", conn)
            conn.close()

            total_spend = df_transactions.groupby('hshd_num')['spend'].sum().reset_index()
            total_spend.rename(columns={'spend': 'total_spend'}, inplace=True)

            df_merged = pd.merge(df_households, total_spend, how='left', on='hshd_num')
            df_merged['total_spend'].fillna(0, inplace=True)

            # Define churn: customers who spent < 500 are at risk
            df_merged['churn'] = df_merged['total_spend'].apply(lambda x: 1 if x < 500 else 0)

            # Use only available features
            available_features = ['age_range', 'marital_status', 'income_range', 'homeowner_desc', 'total_spend']
            existing_features = [f for f in available_features if f in df_merged.columns]

            df_features = pd.get_dummies(df_merged[existing_features])
            labels = df_merged['churn']

            if len(df_features) < 10:
                st.warning("Not enough data available to train Churn Prediction model.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(df_features, labels, test_size=0.3, random_state=42)

                churn_model = RandomForestClassifier()
                churn_model.fit(X_train, y_train)

                churn_accuracy = churn_model.score(X_test, y_test)
                st.success(f"Churn Prediction Model Accuracy: {churn_accuracy*100:.2f}%")

                churn_importances = pd.Series(churn_model.feature_importances_, index=X_train.columns)
                st.subheader("Top Factors Influencing Churn")
                st.bar_chart(churn_importances.sort_values(ascending=False).head(5))

        except Exception as e:
            st.error(f"Churn Model Error: {e}")

else:
    st.warning("🔒 Please login to continue.")

