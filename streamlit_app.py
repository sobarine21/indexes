import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(page_title="Accurate Enforcement Index Generator", layout="wide")
st.title("📊 Accurate Enforcement Index Generator")

st.markdown("""
Upload your **Constituents** file and up to two **Enforcement** files. The application will calculate a unified Enforcement Score for each company.
- **File 2A (Events):** Contains individual complaint records. Used for calculating total, unresolved, and recent complaints.
- **File 2C (Summary):** Contains company-level summary data. Used as a fallback for total complaints and for pending complaints / shareholder numbers.

**Each company will appear only once in the final output.**
""")
st.markdown("---")


# --- Helper Functions ---

def load_file(uploaded_file):
    """Loads an uploaded file (CSV or Excel) into a pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
        return None

def find_company_column(df):
    """Finds the first column in a DataFrame that contains 'COMPANY' in its name."""
    for col in df.columns:
        if "COMPANY" in col.upper():
            return col
    return None

def to_numeric_safe(series):
    """Safely converts a pandas Series to a numeric type, replacing common non-numeric values with 0."""
    return pd.to_numeric(series.replace(['-', '–', 'nan', 'NaN', 'None', None], 0), errors="coerce").fillna(0).astype(int)

def clean_and_standardize_headers(df):
    """Standardizes DataFrame column headers to uppercase and strips whitespace."""
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def minmax_scale(series):
    """Applies Min-Max normalization to a pandas Series."""
    arr = series.astype(float).values
    # Avoid division by zero if all values are the same
    if arr.max() == arr.min():
        return np.zeros_like(arr, dtype=float)
    return (arr - arr.min()) / (arr.max() - arr.min())

# --- 1. File Uploads ---
st.header("1. Upload Files")
col1, col2, col3 = st.columns(3)

with col1:
    constituents_file = st.file_uploader(
        "Upload Constituents File",
        type=["xlsx", "xls", "csv"],
        key="constituents"
    )
with col2:
    enforcement_file_2a = st.file_uploader(
        "Upload Enforcement File 2A (Events)",
        type=["xlsx", "xls", "csv"],
        key="enforcement2a"
    )
with col3:
    enforcement_file_2c = st.file_uploader(
        "Upload Enforcement File 2C (Summary)",
        type=["xlsx", "xls", "csv"],
        key="enforcement2c"
    )

df_const, df_2a, df_2c = None, None, None

# --- Data Loading and Preprocessing ---

if constituents_file:
    df_const = load_file(constituents_file)
    if df_const is not None:
        df_const = clean_and_standardize_headers(df_const)
        # Standardize company column name for easier merging
        company_col_const = find_company_column(df_const)
        if company_col_const:
            df_const.rename(columns={company_col_const: "COMPANY"}, inplace=True)
            df_const["COMPANY"] = df_const["COMPANY"].astype(str).str.upper().str.strip()

if enforcement_file_2a:
    df_2a = load_file(enforcement_file_2a)
    if df_2a is not None:
        df_2a = clean_and_standardize_headers(df_2a)
        company_col_2a = find_company_column(df_2a)
        if company_col_2a:
            df_2a.rename(columns={company_col_2a: "COMPANY"}, inplace=True)
            df_2a["COMPANY"] = df_2a["COMPANY"].astype(str).str.upper().str.strip()
        
        # Find date column robustly
        date_col_2a = next((c for c in df_2a.columns if "DATE OF REC" in c), None)
        if date_col_2a:
            df_2a[date_col_2a] = pd.to_datetime(df_2a[date_col_2a], errors="coerce")
        else:
            st.warning("Could not find a 'Date of Receipt' column in File 2A. 'Recent' complaints will be 0.")

if enforcement_file_2c:
    df_2c = load_file(enforcement_file_2c)
    if df_2c is not None:
        df_2c = clean_and_standardize_headers(df_2c)
        company_col_2c = find_company_column(df_2c)
        if company_col_2c:
            df_2c.rename(columns={company_col_2c: "COMPANY"}, inplace=True)
            df_2c["COMPANY"] = df_2c["COMPANY"].astype(str).str.upper().str.strip()
        
        # Convert all potentially numeric columns safely
        num_cols = [
            "NO. OF SHAREHOLDERS", "RECEIVED", "REDRESSED THROUGH EXCHANGE",
            "NON-ACTIONABLE~", "ADVISED / OPTED FOR ARBITRATION", "PENDING FOR REDRESSAL WITH EXCHANGE"
        ]
        for c in num_cols:
            if c in df_2c.columns:
                # CRITICAL FIX: Re-assign the result of the conversion
                df_2c[c] = to_numeric_safe(df_2c[c])

# Display previews
if df_const is not None:
    st.subheader("Constituents Preview")
    st.dataframe(df_const.head(), use_container_width=True)
if df_2a is not None:
    st.subheader("File 2A (Events) Preview")
    st.dataframe(df_2a.head(), use_container_width=True)
if df_2c is not None:
    st.subheader("File 2C (Summary) Preview")
    st.dataframe(df_2c.head(), use_container_width=True)

st.markdown("---")


# --- Main Calculation Logic ---
if df_const is not None and (df_2a is not None or df_2c is not none):
    st.header("2. Configure Metric Weights")

    wcols = st.columns(5)
    w_total = wcols[0].number_input("Weight: Total Complaints", 0.0, 1.0, 0.4, 0.05)
    w_unresolved = wcols[1].number_input("Weight: Unresolved Complaints", 0.0, 1.0, 0.2, 0.05)
    w_recent = wcols[2].number_input("Weight: Recent Complaints (1yr)", 0.0, 1.0, 0.2, 0.05)
    w_pending = wcols[3].number_input("Weight: Pending Complaints", 0.0, 1.0, 0.1, 0.05)
    w_shareholders = wcols[4].number_input("Weight: Shareholder Base (Bonus)", 0.0, 1.0, 0.1, 0.05)
    st.caption("Higher weights mean a greater impact on the score. All metrics except Shareholder Base are penalties (i.e., they lower the score).")
    
    st.markdown("---")
    st.header("3. Generate Enforcement Index")

    if st.button("Calculate Now", type="primary"):
        with st.spinner("Calculating..."):
            # Start with the base list of companies
            merged = df_const.copy()

            # --- Calculate Metrics from Source Files ---
            
            # From File 2A (Events)
            if df_2a is not None and "COMPANY" in df_2a.columns:
                # Total Complaints (Primary Source)
                total_counts = df_2a.groupby("COMPANY").size().rename("TOTAL_COMPLAINTS_2A")
                merged = merged.merge(total_counts, on="COMPANY", how="left")
                
                # Unresolved Complaints
                if "STATUS" in df_2a.columns:
                    unresolved = df_2a[~df_2a["STATUS"].str.contains("RESOLVED", case=False, na=False)]
                    unresolved_counts = unresolved.groupby("COMPANY").size().rename("UNRESOLVED")
                    merged = merged.merge(unresolved_counts, on="COMPANY", how="left")
                
                # Recent Complaints (within last 365 days)
                date_col_2a = next((c for c in df_2a.columns if "DATE OF REC" in c), None)
                if date_col_2a:
                    one_year_ago = pd.Timestamp.now() - pd.Timedelta(days=365)
                    recent = df_2a[df_2a[date_col_2a] >= one_year_ago]
                    recent_counts = recent.groupby("COMPANY").size().rename("RECENT")
                    merged = merged.merge(recent_counts, on="COMPANY", how="left")

            # From File 2C (Summary)
            if df_2c is not None and "COMPANY" in df_2c.columns:
                metrics_2c = ["RECEIVED", "PENDING FOR REDRESSAL WITH EXCHANGE", "NO. OF SHAREHOLDERS"]
                cols_to_merge = ["COMPANY"] + [c for c in metrics_2c if c in df_2c.columns]
                merged = merged.merge(df_2c[cols_to_merge], on="COMPANY", how="left")
            
            # --- Consolidate Metrics ---
            
            # Logic: Use complaints from 2A if available, otherwise use 'RECEIVED' from 2C.
            if "TOTAL_COMPLAINTS_2A" in merged.columns and "RECEIVED" in merged.columns:
                merged["TOTAL_COMPLAINTS"] = merged["TOTAL_COMPLAINTS_2A"].fillna(merged["RECEIVED"])
            elif "TOTAL_COMPLAINTS_2A" in merged.columns:
                merged["TOTAL_COMPLAINTS"] = merged["TOTAL_COMPLAINTS_2A"]
            elif "RECEIVED" in merged.columns:
                merged["TOTAL_COMPLAINTS"] = merged["RECEIVED"]
            
            # Rename for consistency and fill NaNs
            if "PENDING FOR REDRESSAL WITH EXCHANGE" in merged.columns:
                merged.rename(columns={"PENDING FOR REDRESSAL WITH EXCHANGE": "PENDING"}, inplace=True)
            if "NO. OF SHAREHOLDERS" in merged.columns:
                merged.rename(columns={"NO. OF SHAREHOLDERS": "SHAREHOLDERS"}, inplace=True)

            metric_cols = ["TOTAL_COMPLAINTS", "UNRESOLVED", "RECENT", "PENDING", "SHAREHOLDERS"]
            for col in metric_cols:
                if col not in merged.columns:
                    merged[col] = 0 # Ensure column exists before filling NaN
            merged[metric_cols] = merged[metric_cols].fillna(0).astype(int)

            # --- Normalize Metrics (0 to 1) ---
            scaled_metrics = pd.DataFrame(index=merged.index)
            for col in metric_cols:
                scaled_metrics[f"scaled_{col}"] = minmax_scale(merged[col])

            # --- Calculate Final Score ---
            # Score = 100 * (1 - (sum of weighted penalty metrics) + (weighted bonus metric))
            merged["ENFORCEMENT_SCORE"] = 100 * (
                1
                - w_total * scaled_metrics["scaled_TOTAL_COMPLAINTS"]
                - w_unresolved * scaled_metrics["scaled_UNRESOLVED"]
                - w_recent * scaled_metrics["scaled_RECENT"]
                - w_pending * scaled_metrics["scaled_PENDING"]
                + w_shareholders * scaled_metrics["scaled_SHAREHOLDERS"]
            )
            
            # Ensure score is within the 0-100 range
            merged["ENFORCEMENT_SCORE"] = merged["ENFORCEMENT_SCORE"].clip(lower=0, upper=100)
            merged["RANK"] = merged["ENFORCEMENT_SCORE"].rank(method='min', ascending=False).astype(int)
            
            # --- Display Results ---
            st.subheader("Enforcement Index Results")
            display_cols = [
                "RANK", "COMPANY", "ENFORCEMENT_SCORE", "TOTAL_COMPLAINTS", 
                "UNRESOLVED", "RECENT", "PENDING", "SHAREHOLDERS"
            ]
            # Add other descriptive columns from constituents if they exist
            for col in ["INDUSTRY", "SYMBOL", "SERIES", "ISIN"]:
                if col in merged.columns:
                    display_cols.insert(2, col)

            result_df = merged[[c for c in display_cols if c in merged.columns]].sort_values("RANK")
            st.dataframe(result_df, use_container_width=True, height=600)

            # --- Download Options ---
            st.header("4. Download Results")
            
            # Excel Download
            excel_buffer = io.BytesIO()
            result_df.to_excel(excel_buffer, index=False, engine="openpyxl")
            st.download_button(
                label="📥 Download as Excel",
                data=excel_buffer.getvalue(),
                file_name="enforcement_index_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # CSV Download
            csv_buffer = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download as CSV",
                data=csv_buffer,
                file_name="enforcement_index_results.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload the **Constituents** file and at least one **Enforcement** file to proceed.")

st.markdown("---")
st.markdown("Developed for accurate, unified company scoring.")
