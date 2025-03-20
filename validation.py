import pandas as pd
import streamlit as st
import re
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from fuzzywuzzy import process
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Constants
# PROFILE_FILE = "user_profiles.json"

# Load or create user profiles
# def load_profiles():
#     try:
#         with open(PROFILE_FILE, "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {}

# def save_profiles(profiles):
#     with open(PROFILE_FILE, "w") as f:
#         json.dump(profiles, f, indent=4)
# File upload section
st.title("📊Data Cleaning & Validation Dashboard")
# profiles = load_profiles()
# profile_name = st.text_input("Enter Profile Name (Optional)")
# if profile_name and profile_name in profiles:
#     selected_columns = profiles[profile_name]['selected_columns']
#     user_rules = profiles[profile_name]['user_rules']
# else:
#     selected_columns, user_rules = [], {}
file = st.file_uploader("📂 Upload a File (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"])

if file:
    file_ext = file.name.split(".")[-1]

    # Load Data
    if file_ext == "csv":
        df = pd.read_csv(file)
    elif file_ext in ["xlsx", "xls"]:
        df = pd.read_excel(file)
    elif file_ext == "json":
        df = pd.read_json(file)

    st.write("### 🏷 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Select columns for cleaning
    all_columns = list(df.columns)
    selected_columns = st.multiselect("📝 Select Columns to Clean", all_columns, default=all_columns)

    # **1️⃣ Detect Column Types & Auto-Apply Rules**
    def infer_column_types(df):
        """Automatically infer column types based on regex patterns."""
        validation_rules = {
            "integer": r"^\d+$",
            "float": r"^\d+(\.\d{1,2})?$",
            "email": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$",
            "phone": r"^\+?[0-9]{10,15}$",
            "date": r"^\d{4}-\d{2}-\d{2}$",
            "text": r"^[A-Za-z\s]+$"
        }
        
        inferred_validations = {}

        for col in df.columns:
            sample_values = df[col].dropna().astype(str).head(10)

            if sample_values.str.match(validation_rules["email"]).all():
                inferred_validations[col] = validation_rules["email"]
            elif sample_values.str.match(validation_rules["phone"]).all():
                inferred_validations[col] = validation_rules["phone"]
            elif sample_values.str.match(validation_rules["date"]).all():
                inferred_validations[col] = validation_rules["date"]
            elif sample_values.str.match(validation_rules["integer"]).all():
                inferred_validations[col] = validation_rules["integer"]
            elif sample_values.str.match(validation_rules["float"]).all():
                inferred_validations[col] = validation_rules["float"]
            else:
                inferred_validations[col] = validation_rules["text"]  # Default to text

        return inferred_validations

    inferred_rules = infer_column_types(df)

    st.write("### 🔎 Auto-Detected Validation Rules")
    user_rules = {}
    for col in selected_columns:
        user_rules[col] = st.text_input(f"Validation rule for {col} (Regex)", value=inferred_rules.get(col, ""))

    # **2️⃣ Machine Learning-Based Column Classification**
    @st.cache_data
    def train_column_classifier():
        """Train ML model to classify column types."""
        data_samples = [
            ("john.doe@email.com", "email"),
            ("1234567890", "phone"),
            ("2023-05-12", "date"),
            ("98765", "integer"),
            ("Manager", "text"),
        ]

        df_train = pd.DataFrame(data_samples, columns=["sample", "type"])
        vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
        X_train = vectorizer.fit_transform(df_train["sample"])
        y_train = df_train["type"]

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        return vectorizer, clf

    vectorizer, clf = train_column_classifier()

    def predict_column_type(values):
        X_test = vectorizer.transform(values.astype(str))
        return clf.predict(X_test)

    # **3️⃣ AI-Based Auto-Correction**
    def correct_email(email):
        domain_suggestions = ["gmail.com", "yahoo.com", "outlook.com"]
        username, domain = email.split("@") if "@" in email else (email, "")
        closest_match = process.extractOne(domain, domain_suggestions)
        return f"{username}@{closest_match[0]}" if closest_match else email

    # **4️⃣ Suggest Corrections Instead of Rejecting**
    def clean_data(df, selected_columns, user_rules):
        report = {}
        progress_messages = []

        if selected_columns:
            df = df[selected_columns]

        # Duplicate Management
        duplicate_rows = df[df.duplicated()]
        df.drop_duplicates(inplace=True)
        report['duplicates_removed'] = len(duplicate_rows)
        progress_messages.append("✔ Duplicates handled.")

        # Missing Value Handling
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        progress_messages.append("✔ Missing values handled.")

        # Email Validation & Auto-Correction
        for col in df.columns:
            if "email" in col.lower():
                df[f'{col}_valid'] = df[col].astype(str).apply(lambda x: bool(re.match(user_rules[col], x)) if pd.notna(x) else False)
                df[f'{col}_corrected'] = df[col].astype(str).apply(correct_email)

        # Date Validation
        date_columns = [col for col in df.columns if "date" in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # ML-Based Column Type Classification
        column_predictions = predict_column_type(df[selected_columns].astype(str).head(10))
        report["ml_predicted_types"] = dict(zip(selected_columns, column_predictions))

        report['final_row_count'] = len(df)
        
        # Ensure only numeric values are summed
        numeric_report_values = [v for v in report.values() if isinstance(v, (int, float))]
        
        # Calculate data quality score using only numeric values
        data_quality_score = max(0, 100 - (sum(numeric_report_values) / (len(df) + 1) * 100))


        for message in progress_messages:
            st.write(message)

        return df, report, data_quality_score

    # Clean Data
    cleaned_df, report, data_quality_score = clean_data(df, selected_columns, user_rules)

    # Show Report
    st.write("### 📑 Cleaning Report")
    for key, value in report.items():
        st.write(f"✔ {key.replace('_', ' ').capitalize()}: {value}")
    st.write(f"📊 **Data Quality Score:** {data_quality_score:.2f}%")

    # Download Cleaned Data
    cleaned_file = "cleaned_data.csv"
    if not cleaned_df.empty:
        cleaned_df.to_csv(cleaned_file, index=False)
        st.download_button(label="📥 Download Cleaned Data", data=open(cleaned_file, "rb"), file_name="cleaned_data.csv", mime="text/csv")

  
    # Visualization
    st.write("### 📊 Data Cleaning Summary")
    
    # Ensure only numeric values are used for plotting
    numeric_report = {k: v for k, v in report.items() if isinstance(v, (int, float))}
    
    # Check if there are valid values to plot
    if numeric_report:
        fig, ax = plt.subplots()
        sns.barplot(x=list(numeric_report.keys()), y=list(numeric_report.values()), ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("⚠️ No valid numeric data available for visualization.")

