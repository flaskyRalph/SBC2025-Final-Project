
import streamlit  as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta, datetime
import hashlib
from sqlalchemy import create_engine, String
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Mapped, mapped_column


# --- SQLAlchemy ORM Setup ---
DB_USER = 'root'  
DB_PASS = ''     
DB_HOST = 'localhost'
DB_NAME = 'retail_users_db'
DB_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'tbl_users'
    username: Mapped[str] = mapped_column(String(255), primary_key=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    date_created: Mapped[str] = mapped_column(String(255), nullable=True, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# --- User Log Table ---
class UserLog(Base):
    __tablename__ = 'tbl_user_logs'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp: Mapped[str] = mapped_column(String(255), nullable=False, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def log_user_action(username, action):
    create_users_table()  # Ensures tables exist
    session = SessionLocal()
    log = UserLog(username=username, action=action, timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    session.add(log)
    session.commit()
    session.close()

def create_users_table():
    Base.metadata.create_all(engine)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    create_users_table()
    session = SessionLocal()
    user = session.query(User).filter_by(username=username).first()
    if user:
        session.close()
        return False, "Username already exists."
    hashed = hash_password(password)
    new_user = User(username=username, password=hashed)
    session.add(new_user)
    session.commit()
    session.close()
    return True, "Registration successful."

def authenticate_user(username, password):
    create_users_table()
    session = SessionLocal()
    user = session.query(User).filter_by(username=username).first()
    session.close()
    if user and user.password == hash_password(password):
        log_user_action(username, "login")
        return True
    return False

def login_page():
    st.title("User Login")
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if authenticate_user(login_username, login_password):
                st.session_state["logged_in"] = True
                st.session_state["username"] = login_username
                st.success("Login successful!")
                return()
            else:
                st.error("Invalid username or password.")
    with register_tab:
        reg_username = st.text_input("New Username", key="reg_username")
        reg_password = st.text_input("New Password", type="password", key="reg_password")
        reg_password2 = st.text_input("Confirm Password", type="password", key="reg_password2")
        if st.button("Register"):
            if reg_password != reg_password2:
                st.error("Passwords do not match.")
            elif not reg_username or not reg_password:
                st.error("Please fill all fields.")
            else:
                ok, msg = register_user(reg_username, reg_password)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False


# --- Logout Button ---
def logout():
    username = st.session_state.get("username", "")
    if username:
        log_user_action(username, "logout")
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.success("Logged out successfully.")
    return

if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# Show logout button in sidebar if logged in
if st.session_state.get("logged_in", False):
    if st.sidebar.button("Logout"):
        logout()

# --- Page Config ---
st.set_page_config(page_title="🛍️ Retail Inventory Forecasting", layout="wide")
st.title("🛍️ Retail Inventory Forecasting Dashboard")

# --- Load Data ---
def load_data():
    df = pd.read_csv("retail_store_inventory.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

df = load_data()


# --- Sidebar ---
st.sidebar.header("📊 Filter Options")

# Show dataset date range
start_date = df['Date'].min()
end_date = df['Date'].max()
st.sidebar.markdown(f"**Dataset Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# Category selection
categories = df['Category'].unique().tolist()
selected_category = st.sidebar.selectbox("Select Category", categories)

# Region selection
regions = df['Region'].unique().tolist()
selected_region = st.sidebar.selectbox("Select Region", regions)







# Forecast months as a selectbox with month labels
future_months = 12
future_month_labels = []
last_date = end_date
for i in range(1, future_months + 1):
    next_month = (last_date + pd.DateOffset(months=i)).replace(day=1)
    label = next_month.strftime('%B %Y')
    future_month_labels.append(label)
selected_months_label = st.sidebar.selectbox("Forecast up to month", future_month_labels, index=2)
forecast_months = future_month_labels.index(selected_months_label) + 1

# --- Filtered Data ---
df_filtered = df[(df['Category'] == selected_category) & (df['Region'] == selected_region)]
df_filtered = df_filtered.sort_values(by='Date')

if df_filtered.empty:
    st.warning("No data available for the selected category and region.")
    st.stop()


# --- Feature Engineering ---
df_model = df_filtered.copy()
columns_to_encode = ["Seasonality", "Weather Condition", "Region", "Category"]
Lbl = LabelEncoder()
df_model[columns_to_encode] = df_model[columns_to_encode].apply(Lbl.fit_transform)
df_model['Month'] = df_model['Date'].dt.month
df_model['Day'] = df_model['Date'].dt.day

df_model = df_model.drop(["Date", "Store ID", "Product ID"], axis=1)
df_model.dropna(inplace=True)

# --- Forecast Target Selection ---
target_options = [col for col in df_model.columns]
default_target = "Price" if "Price" in target_options else target_options[0]
selected_target = st.sidebar.selectbox("Select column to forecast (target)", target_options, index=target_options.index(default_target))


# --- Model Training ---
st.subheader(f"📈 Forecasting {selected_target} for {selected_category} in {selected_region}")
# Use all numeric columns except the target as features
feature_candidates = [col for col in df_model.columns if col != selected_target]
X = df_model[feature_candidates]
y = df_model[selected_target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

# --- Evaluation ---
st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")

# --- Feature Importance ---
st.subheader("🔍 Feature Importance")
feature_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)
st.bar_chart(feature_imp.set_index("Feature"))

# --- Plot Actual vs Predicted ---
st.subheader("📊 Actual vs Predicted Price")
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(range(len(y_test)), y_test.values, label="Actual", linewidth=2)
ax1.plot(range(len(y_test)), preds, label="Predicted", linestyle='--')
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- Forecast Future Values ---
st.subheader(f"🔮 Forecast Future {selected_target}")
if st.button("Generate Forecast"):
    last_row = df_model.iloc[-1:].copy()
    future_predictions = []
    future_dates = []
    current_date = df_filtered['Date'].max()

    for i in range(forecast_months):
        current_date += pd.DateOffset(months=1)
        future_dates.append(current_date.strftime('%Y-%m'))

        # Update month and day in the input row
        last_row['Month'] = current_date.month
        last_row['Day'] = current_date.day

        # Optionally, simulate changes in other features for more realistic forecasting
        # Here, we'll add a small random walk to all numeric features except the target
        for col in feature_candidates:
            if col not in ['Month', 'Day']:
                last_row[col] = last_row[col] * np.random.uniform(0.98, 1.02)

        next_input = last_row[feature_candidates].values
        next_input_scaled = scaler.transform(next_input)
        next_pred = model.predict(next_input_scaled)[0]

        # Update last_row's target to the predicted value for next iteration (optional, for chained forecasting)
        last_row[selected_target] = next_pred

        future_predictions.append(next_pred)

    # --- Plot Forecast ---
    forecast_df = pd.DataFrame({"Month": future_month_labels[:forecast_months], f"Forecasted {selected_target}": future_predictions})
    st.line_chart(forecast_df.set_index("Month"))
    st.write(forecast_df)
