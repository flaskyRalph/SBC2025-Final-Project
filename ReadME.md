# 🛍️ Retail Inventory Forecasting Dashboard

A **machine learning-powered web application** for forecasting retail product prices using historical inventory, sales, and market data. Built with **Streamlit**, this app includes secure user authentication, category/region-based filtering, data visualization, and future price forecasting using **Random Forest Regression**.

## 🚀 Features

- User registration and secure login (passwords hashed via SHA256)
- Data filtering by Category and Region
- MAE & RMSE performance metrics
- Actual vs Predicted Price visualization
- Feature importance bar chart
- 12-month future price forecasting
- Interactive Streamlit-based UI
- MySQL backend integration with SQLAlchemy ORM

## 🧠 Technologies & Libraries Used

- Python
- [Streamlit](https://streamlit.io)
- Pandas & NumPy
- Scikit-learn
- Matplotlib
- SQLAlchemy ORM
- MySQL (via MySQL Connector)

## 🗃️ Database Setup

Ensure MySQL is running and has a database named `retail_users`.

SQLAlchemy connection string used:


## Minimal Required Installs

pip install streamlit pandas numpy matplotlib scikit-learn SQLAlchemy mysql-connector-python
