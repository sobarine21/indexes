import streamlit as st
import pandas as pd
import numpy as np
import io

# --- Page Configuration ---
st.set_page_config(page_title="Advanced Enforcement Index Generator", layout="wide")
st.title("📈 Advanced Enforcement Index Generator")

st.markdown("""
This tool calculates a sophisticated **Enforcement Score** for each company based on multiple data sources. 
It ensures **no duplicate companies** appear in the output and provides advanced options for customized, accurate analysis.
""")

# --- Helper Functions ---

def load_file(uploaded_file):
    """Loads an uploaded file (CSV or Excel) into a pandas DataFrame."""
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file) if uploaded_file.name.lower().endswith('.csv') else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading {getattr(uploaded_file, 'name', 'file')}: {e}")
        return None

def find_company_column(df):
    """Finds the first column in a DataFrame that contains 'COMPANY' in its name."""
    return next((col for col in df.columns if "COMPANY" in str(col).upper()), None)

def to_numeric_safe(series):
    """Safely converts a pandas Series to numeric, replacing common non-numeric values with 0."""
    return pd.to_numeric(series.replace(['-', '–', '—', 'nan', 'NaN', 'None', None, ""], 0), errors="coerce").fillna(0)

def clean_and_standardize_headers(df):
    """Standardizes DataFrame column headers to uppercase and strips whitespace."""
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def normalize_data(df, columns, method='Min-Max Scaling'):
    """Applies the selected normalization method to the specified columns."""
    normalized_df = pd.DataFrame(index=df.index)
    for col in columns:
        series = df[col].astype(float)
        if method == 'Min-Max Scaling':
            # Avoid division by zero if all values are the same
            if series.max() == series.min():
                normalized_df[f"scaled_{col}"] = 0.0
            else:
                normalized_df[f"scaled_{col}"] = (series - series.min()) / (series.max() - series.min())
        elif method == 'Z-Score Standardization':
            # Avoid division by zero if standard deviation is zero
            std = series.std()
            if std == 0:
                normalized_df[f"scaled_{col}"] = 0.0
            else:
                normalized_df[f"scaled_{col}"] = (series - series.mean()) / std
        else:
            normalized_df[f"scaled_{col}"] = series  # fallback (shouldn't happen)
    return normalized_df

def detect_date_of_receipt_column(df):
    """Finds the 2A date column (handles 'DATE OF RECIEPT' typo and 'DATE OF RECEIPT')."""
    candidates = [c for c in df.columns if "DATE OF REC" in c]  # catches RECIEPT/RECEIPT
    if candidates:
        return candidates[0]
    # secondary heuristic
    candidates = [c for c in df.columns if "DATE" in c and "REC" in c]
    return candidates[0] if candidates else None

# --- Sidebar for Advanced Settings ---
st.sidebar.header("⚙️ Advanced Settings")
normalization_method = st.sidebar.radio(
    "1. Normalization Method",
    ["Min-Max Scaling", "Z-Score Standardization"],
    help="**Min-Max:** Scales values to a [0, 1] range. Good for general use. **Z-Score:** Standardizes values based on mean and standard deviation, useful for handling outliers."
)
shareholder_logic = st.sidebar.radio(
    "2. Shareholder Data Logic",
    ["Use as Bonus Factor", "Calculate Complaint Rate (Penalty)"],
    help="**Bonus Factor:** Larger companies get a score boost. **Complaint Rate:** Calculates complaints per shareholder, penalizing companies with a higher rate regardless of size."
)

# --- 1. File Uploads ---
st.header("1. Upload Files")
col1, col2, col3 = st.columns(3)
constituents_file = col1.file_uploader("Upload Constituents File", type=["xlsx", "xls", "csv"])
enforcement_file_2a = col2.file_uploader("Upload File 2A (Events)", type=["xlsx", "xls", "csv"])
enforcement_file_2c = col3.file_uploader("Upload File 2C (Summary)", type=["xlsx", "xls", "csv"])

# --- Data Loading and Preprocessing ---
df_const = load_file(constituents_file)
df_2a = load_file(enforcement_file_2a)
df_2c = load_file(enforcement_file_2c)

if df_const is not None:
    df_const = clean_and_standardize_headers(df_const)
    company_col_const = find_company_column(df_const)
    if company_col_const:
        df_const.rename(columns={company_col_const: "COMPANY"}, inplace=True)
    if "COMPANY" in df_const.columns:
        df_const["COMPANY"] = df_const["COMPANY"].astype(str).str.upper().str.strip()

if df_2a is not None:
    df_2a = clean_and_standardize_headers(df_2a)
    company_col_2a = find_company_column(df_2a)
    if company_col_2a:
        df_2a.rename(columns={company_col_2a: "COMPANY"}, inplace=True)
    if "COMPANY" in df_2a.columns:
        df_2a["COMPANY"] = df_2a["COMPANY"].astype(str).str.upper().str.strip()
    # STATUS normalization (missing -> RESOLVED)
    if "STATUS" not in df_2a.columns:
        df_2a["STATUS"] = "RESOLVED"
    df_2a["STATUS"] = df_2a["STATUS"].astype(str).str.strip().str.upper().replace({"": "RESOLVED"})
    # Parse date of receipt robustly
    date_col_2a = detect_date_of_receipt_column(df_2a)
    if date_col_2a:
        df_2a[date_col_2a] = pd.to_datetime(df_2a[date_col_2a], errors="coerce")

if df_2c is not None:
    df_2c = clean_and_standardize_headers(df_2c)
    company_col_2c = find_company_column(df_2c)
    if company_col_2c:
        df_2c.rename(columns={company_col_2c: "COMPANY"}, inplace=True)
    if "COMPANY" in df_2c.columns:
        df_2c["COMPANY"] = df_2c["COMPANY"].astype(str).str.upper().str.strip()

    # Numeric columns (robustly detect shareholders)
    shareholder_col = next((c for c in df_2c.columns if "SHAREHOLDER" in c), "NO. OF SHAREHOLDERS")
    numeric_candidates = ["RECEIVED", "PENDING FOR REDRESSAL WITH EXCHANGE", shareholder_col,
                          "REDRESSED THROUGH EXCHANGE", "NON-ACTIONABLE~", "ADVISED / OPTED FOR ARBITRATION"]
    for c in numeric_candidates:
        if c in df_2c.columns:
            df_2c[c] = to_numeric_safe(df_2c[c]).astype(int)

st.markdown("---")

# --- Main Calculation Logic ---
if df_const is not None:
    st.header("2. Configure Metric Weights")
    
    st.subheader("Core Penalty Weights")
    wcols1 = st.columns(4)
    w_total = wcols1[0].slider("Total Complaints", 0.0, 1.0, 0.40, 0.05)
    w_unresolved = wcols1[1].slider("Unresolved Complaints", 0.0, 1.0, 0.20, 0.05)
    w_recent = wcols1[2].slider("Recent Complaints (1yr)", 0.0, 1.0, 0.20, 0.05)
    w_pending = wcols1[3].slider("Pending Complaints", 0.0, 1.0, 0.10, 0.05)

    st.subheader("Advanced & Relative Metric Weights")
    wcols2 = st.columns(4)
    if shareholder_logic == "Use as Bonus Factor":
        w_shareholders_bonus = wcols2[0].slider("Shareholder Base (Bonus)", 0.0, 1.0, 0.10, 0.05, help="Higher shareholder count increases the score.")
        w_complaint_rate = 0.0
        w_pending_ratio = wcols2[1].slider("Pending Ratio (Penalty)", 0.0, 1.0, 0.10, 0.05, help="Higher % of pending complaints decreases the score.")
    else:
        w_complaint_rate = wcols2[0].slider("Complaint Rate (Penalty)", 0.0, 1.0, 0.20, 0.05, help="Higher complaints per shareholder decreases the score.")
        w_pending_ratio = wcols2[1].slider("Pending Ratio (Penalty)", 0.0, 1.0, 0.10, 0.05)
        w_shareholders_bonus = 0.0

    st.markdown("---")
    st.header("3. Generate Enforcement Index")

    if st.button("Calculate Index", type="primary", use_container_width=True):
        with st.spinner("Processing data..."):
            # 1) Start with the base list of companies
            merged = df_const.copy()

            # 2) Metrics from 2A
            if df_2a is not None and "COMPANY" in df_2a.columns:
                # Total complaints per company (2A)
                total_counts = df_2a.groupby("COMPANY").size().rename("TOTAL_COMPLAINTS_2A")
                merged = merged.merge(total_counts, on="COMPANY", how="left")

                # Unresolved complaints per company (STATUS != RESOLVED)
                unresolved = df_2a[df_2a["STATUS"] != "RESOLVED"]
                unresolved_counts = unresolved.groupby("COMPANY").size().rename("UNRESOLVED")
                merged = merged.merge(unresolved_counts, on="COMPANY", how="left")

                # Recent complaints (last 365 days) by date of receipt
                date_col_2a = detect_date_of_receipt_column(df_2a)
                if date_col_2a:
                    recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
                    recent_mask = pd.to_datetime(df_2a[date_col_2a], errors="coerce") >= recent_cutoff
                    recent = df_2a[recent_mask]
                    recent_counts = recent.groupby("COMPANY").size().rename("RECENT")
                    merged = merged.merge(recent_counts, on="COMPANY", how="left")

            # 3) Metrics from 2C
            if df_2c is not None and "COMPANY" in df_2c.columns:
                shareholder_col = next((c for c in df_2c.columns if "SHAREHOLDER" in c), "NO. OF SHAREHOLDERS")
                cols_to_merge = ["COMPANY", "RECEIVED", "PENDING FOR REDRESSAL WITH EXCHANGE", shareholder_col]
                cols_to_merge = [c for c in cols_to_merge if c in df_2c.columns]
                merged = merged.merge(df_2c[cols_to_merge], on="COMPANY", how="left")

            # 4) Consolidate & Clean
            # TOTAL_COMPLAINTS: prefer 2A counts; if missing, fallback to 2C RECEIVED
            if "TOTAL_COMPLAINTS_2A" in merged.columns and "RECEIVED" in merged.columns:
                merged["TOTAL_COMPLAINTS"] = merged["TOTAL_COMPLAINTS_2A"].fillna(merged["RECEIVED"])
            elif "TOTAL_COMPLAINTS_2A" in merged.columns:
                merged["TOTAL_COMPLAINTS"] = merged["TOTAL_COMPLAINTS_2A"]
            elif "RECEIVED" in merged.columns:
                merged["TOTAL_COMPLAINTS"] = merged["RECEIVED"]
            else:
                merged["TOTAL_COMPLAINTS"] = 0

            # Rename to standard names
            rename_dict = {"PENDING FOR REDRESSAL WITH EXCHANGE": "PENDING"}
            sh_col_in_merged = next((c for c in merged.columns if "SHAREHOLDER" in c), None)
            if sh_col_in_merged:
                rename_dict[sh_col_in_merged] = "SHAREHOLDERS"
            merged.rename(columns=rename_dict, inplace=True)

            # Ensure base metric columns exist and are numeric
            base_metric_cols = ["TOTAL_COMPLAINTS", "UNRESOLVED", "RECENT", "PENDING", "SHAREHOLDERS"]
            for col in base_metric_cols:
                if col not in merged.columns:
                    merged[col] = 0
                merged[col] = to_numeric_safe(merged[col]).astype(int)

            # 5) Aggregate duplicates -> one row per company (sum metrics; keep first for others)
            agg_dict = {col: 'first' for col in merged.columns if col not in base_metric_cols and col != 'COMPANY'}
            agg_dict.update({col: 'sum' for col in base_metric_cols})
            merged = merged.groupby("COMPANY", as_index=False).agg(agg_dict)

            # 6) Advanced Relative Metrics
            # Complaint rate per 10,000 shareholders (avoid div-by-zero)
            if shareholder_logic == "Calculate Complaint Rate (Penalty)":
                merged["COMPLAINT_RATE"] = np.divide(
                    merged["TOTAL_COMPLAINTS"] * 10000.0,
                    merged["SHAREHOLDERS"],
                    where=merged["SHAREHOLDERS"] != 0,
                    out=np.zeros_like(merged["TOTAL_COMPLAINTS"], dtype=float)
                )
            # Pending ratio (pending / total)
            merged["PENDING_RATIO"] = np.divide(
                merged["PENDING"],
                merged["TOTAL_COMPLAINTS"],
                where=merged["TOTAL_COMPLAINTS"] != 0,
                out=np.zeros_like(merged["PENDING"], dtype=float)
            )

            # 7) Normalize Metrics
            metrics_to_normalize = ["TOTAL_COMPLAINTS", "UNRESOLVED", "RECENT", "PENDING", "PENDING_RATIO"]
            if shareholder_logic == "Use as Bonus Factor":
                metrics_to_normalize.append("SHAREHOLDERS")
            else:
                metrics_to_normalize.append("COMPLAINT_RATE")

            scaled_metrics = normalize_data(merged, metrics_to_normalize, normalization_method)

            # 8) Final Score
            score = 100.0
            score -= w_total * scaled_metrics.get("scaled_TOTAL_COMPLAINTS", 0)
            score -= w_unresolved * scaled_metrics.get("scaled_UNRESOLVED", 0)
            score -= w_recent * scaled_metrics.get("scaled_RECENT", 0)
            score -= w_pending * scaled_metrics.get("scaled_PENDING", 0)
            score -= w_pending_ratio * scaled_metrics.get("scaled_PENDING_RATIO", 0)
            score += w_shareholders_bonus * scaled_metrics.get("scaled_SHAREHOLDERS", 0)
            score -= w_complaint_rate * scaled_metrics.get("scaled_COMPLAINT_RATE", 0)

            merged["ENFORCEMENT_SCORE"] = score
            if normalization_method == 'Min-Max Scaling':
                merged["ENFORCEMENT_SCORE"] = merged["ENFORCEMENT_SCORE"].clip(lower=0, upper=100)

            merged["RANK"] = merged["ENFORCEMENT_SCORE"].rank(method='min', ascending=False).astype(int)

            # 9) Display Results
            st.success("✅ Calculation complete! Each company is now listed only once.")
            display_cols = [
                "RANK", "COMPANY", "ENFORCEMENT_SCORE",
                "INDUSTRY", "SYMBOL", "SERIES", "ISIN",
                "TOTAL_COMPLAINTS", "UNRESOLVED", "RECENT", "PENDING", "PENDING_RATIO", "SHAREHOLDERS"
            ]
            if "COMPLAINT_RATE" in merged.columns:
                display_cols.append("COMPLAINT_RATE")

            result_df = merged[[c for c in display_cols if c in merged.columns]] \
                .sort_values("RANK") \
                .reset_index(drop=True)

            st.dataframe(result_df, use_container_width=True, height=600)

            # 10) Download Options (ensure bytes are passed)
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="Enforcement Index")
            excel_bytes = excel_buffer.getvalue()
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")

            d_col1, d_col2 = st.columns(2)
            d_col1.download_button("📥 Download as Excel", data=excel_bytes, file_name="enforcement_index.xlsx")
            d_col2.download_button("📄 Download as CSV", data=csv_bytes, file_name="enforcement_index.csv")

else:
    st.info("👋 Welcome! Please upload your Constituents file to begin.")
