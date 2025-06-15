import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Custom page config and style
st.set_page_config(page_title="🛍️ Product Reviews Analysis", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #004080;
    }
    .stButton>button {
        background-color: #0072B2;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
csv_path = os.path.join(os.path.dirname(__file__), "reviews.csv")
df = pd.read_csv(csv_path, low_memory=False)

# Introduction
st.title("🛍️ Product Reviews Analysis & Prediction App")
st.markdown("""
Welcome! This interactive dashboard explores product reviews data with visual analytics and enables live predictions of review ratings.
""")

# EDA Section
st.header("🔍 Exploratory Data Analysis")

with st.expander("📄 Show Raw Data"):
    st.dataframe(df.head())

# Use columns to control graph width
col1, col2 = st.columns(2)

with col1:
    st.subheader("⭐ Distribution of Review Ratings")
    if 'reviews.rating' in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df['reviews.rating'].dropna().astype(int).value_counts().sort_index()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Rating Distribution")
        st.pyplot(fig)
    else:
        st.warning("Column 'reviews.rating' not found in dataset.")

with col2:
    st.subheader("🏷️ Average Rating by Brand")
    if 'brand' in df.columns and 'reviews.rating' in df.columns:
        brand_rating = df[['brand', 'reviews.rating']].dropna()
        if not brand_rating.empty:
            avg_rating = brand_rating.groupby("brand")["reviews.rating"].mean().sort_values(ascending=False).head(10)
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            sns.barplot(x=avg_rating.values, y=avg_rating.index, ax=ax2, palette="Oranges_r")
            ax2.set_xlabel("Average Rating")
            ax2.set_ylabel("Brand")
            ax2.set_title("Top 10 Brands")
            st.pyplot(fig2)
        else:
            st.warning("No data available for average ratings by brand.")
    else:
        st.warning("Required columns not found in dataset.")

# Preprocessing
df_model = df.copy()
df_model['reviews.date'] = pd.to_datetime(df_model['reviews.date'], errors='coerce')
df_model = df_model.drop(columns=['reviews.userCity', 'reviews.userProvince', 'reviews.id', 'reviews.date'], errors='ignore')
df_model = df_model[df_model['reviews.rating'].notnull()]

# Encode categorical variables
label_encoders = {}
for col in df_model.select_dtypes(include=['object', 'bool']).columns:
    df_model[col] = df_model[col].astype(str).fillna("Unknown")
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Fill missing and split
X = df_model.drop(columns='reviews.rating')
y = df_model['reviews.rating']
X.fillna(X.median(numeric_only=True), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Section
st.header("📈 Model Performance")
st.metric("🎯 Accuracy", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
st.code(classification_report(y_test, y_pred), language='text')

# Runtime Prediction
st.header("🤖 Predict Review Rating")
st.markdown("""
Adjust the values below and click the button to predict the product review rating.
Only a few key features are needed.
""")

relevant_features = ['brand', 'categories', 'reviews.doRecommend']
user_input = {}
for col in relevant_features:
    if col in df.columns:
        unique_values = df[col].dropna().astype(str).unique()
        user_input[col] = st.selectbox(f"Select {col}", options=sorted(unique_values))

if st.button("🚀 Predict Now"):
    user_df = pd.DataFrame([user_input])

    for col in user_df.columns:
        if col in label_encoders:
            user_df[col] = label_encoders[col].transform(user_df[col].astype(str))

    for col in X.columns:
        if col not in user_df.columns:
            user_df[col] = 0

    user_df = user_df[X.columns]
    user_df_scaled = scaler.transform(user_df)
    prediction = model.predict(user_df_scaled)
    st.success(f"✅ Predicted Review Rating: {prediction[0]}")

# Conclusion
st.header("📌 Conclusion")
st.markdown("""
This app:
- Visually explores product reviews
- Trains a Random Forest model
- Provides real-time predictions for review ratings

🧠 Try changing values above to see prediction results!
""")
