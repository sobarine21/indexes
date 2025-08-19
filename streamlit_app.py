import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Accurate Enforcement Index Generator", layout="wide")
st.title("Accurate Enforcement Index Generator")

st.markdown("""
Upload your *Constituents* file and up to two Enforcement files:
- **File 2A:** Complaints/events (each row = one complaint, with `STATUS`, `DATE OF RECIEPT`, etc.)
- **File 2C:** Company summary (each row = one company, with `RECEIVED`, `NO. OF SHAREHOLDERS`, etc.)

The system merges, lets you customize the Enforcement Index calculation, and ensures all companies have accurate, stable, and comparable enforcement scores.
""")

def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def find_company_column(df):
    for col in df.columns:
        if "COMPANY" in col.upper():
            return col
    return None

def to_numeric_safe(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col].replace(['-', '–', 'nan', 'NaN', 'None', None], 0), errors="coerce").fillna(0).astype(int)
    return df

def clean_and_standardize(df):
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df

def minmax_scale(series):
    arr = series.astype(float).values
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

# --- File Uploads ---
st.header("1. Upload Files")
col1, col2, col3 = st.columns(3)

constituents_file = col1.file_uploader(
    "Upload Constituents File (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="constituents"
)
enforcement_file_2a = col2.file_uploader(
    "Upload Enforcement File 2A: Events (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="enforcement2a"
)
enforcement_file_2c = col3.file_uploader(
    "Upload Enforcement File 2C: Summary (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="enforcement2c"
)

df_const, df_2a, df_2c = None, None, None

if constituents_file:
    df_const = load_file(constituents_file)
    df_const = clean_and_standardize(df_const)
    df_const.rename(columns={
        "COMPANY NAME": "COMPANY",
        "SYMBOL": "SYMBOL",
        "SERIES": "SERIES",
        "ISIN CODE": "ISIN",
        "INDUSTRY": "INDUSTRY"
    }, inplace=True)
    if "COMPANY" in df_const.columns:
        df_const["COMPANY"] = df_const["COMPANY"].astype(str).str.upper().str.strip()

if enforcement_file_2a:
    df_2a = load_file(enforcement_file_2a)
    df_2a = clean_and_standardize(df_2a)
    company_col_2a = find_company_column(df_2a)
    if company_col_2a:
        df_2a[company_col_2a] = df_2a[company_col_2a].astype(str).str.upper().str.strip()
    if "STATUS" not in df_2a.columns:
        df_2a["STATUS"] = "RESOLVED"
    date_col_2a = None
    for c in df_2a.columns:
        if "DATE OF RECIEPT" in c or "DATE OF RECEIPT" in c:
            date_col_2a = c
            break
    if date_col_2a is None:
        df_2a["DATE OF RECIEPT"] = pd.NaT
        date_col_2a = "DATE OF RECIEPT"
    df_2a[date_col_2a] = pd.to_datetime(df_2a[date_col_2a], errors="coerce")

if enforcement_file_2c:
    df_2c = load_file(enforcement_file_2c)
    df_2c = clean_and_standardize(df_2c)
    company_col_2c = find_company_column(df_2c)
    if company_col_2c:
        df_2c[company_col_2c] = df_2c[company_col_2c].astype(str).str.upper().str.strip()
    num_cols = [
        "NO. OF SHAREHOLDERS", "RECEIVED", "REDRESSED THROUGH EXCHANGE",
        "NON-ACTIONABLE~", "ADVISED / OPTED FOR ARBITRATION", "PENDING FOR REDRESSAL WITH EXCHANGE"
    ]
    for c in num_cols:
        to_numeric_safe(df_2c, c)
    for c in list(df_2c.columns):
        if 'DATE' in c and c not in num_cols:
            df_2c.drop(columns=[c], inplace=True, errors="ignore")

if df_const is not None:
    st.subheader("Constituents Preview")
    st.dataframe(df_const.head(10), use_container_width=True)
if df_2a is not None:
    st.subheader("File 2A (Events) Preview")
    st.dataframe(df_2a.head(10), use_container_width=True)
if df_2c is not None:
    st.subheader("File 2C (Summary) Preview")
    st.dataframe(df_2c.head(10), use_container_width=True)

if df_const is not None and (df_2a is not None or df_2c is not None):
    st.header("2. Calculation Settings")

    wcols = st.columns(5)
    w_total = wcols[0].number_input("Weight: Total Complaints", 0.0, 1.0, 0.4)
    w_unresolved = wcols[1].number_input("Weight: Unresolved", 0.0, 1.0, 0.2)
    w_recent = wcols[2].number_input("Weight: Recent (1yr)", 0.0, 1.0, 0.2)
    w_pending = wcols[3].number_input("Weight: Pending", 0.0, 1.0, 0.1)
    w_shareholders = wcols[4].number_input("Weight: Shareholders (positive effect!)", -1.0, 1.0, 0.1)
    st.caption("Weights should sum to 1 or less. Shareholders is positive; others are penalties.")

    st.header("3. Calculation and Ranking")
    merged = df_const.copy()

    # --- Metrics calculation for each company
    n = len(merged)
    metrics = {
        "total": np.zeros(n),
        "unresolved": np.zeros(n),
        "recent": np.zeros(n),
        "pending": np.zeros(n),
        "shareholders": np.zeros(n)
    }

    # Total complaints (from 2A if available, else from 2C)
    if df_2a is not None:
        # Count all complaints by company
        company_col_2a = find_company_column(df_2a)
        total_counts = df_2a.groupby(company_col_2a).size().reset_index(name="TOTAL_COMPLAINTS")
        merged = merged.merge(total_counts, left_on="COMPANY", right_on=company_col_2a, how="left")
        metrics["total"] = merged["TOTAL_COMPLAINTS"].fillna(0).astype(int).values
        merged.drop(columns=["TOTAL_COMPLAINTS", company_col_2a], inplace=True, errors="ignore")
    elif df_2c is not None:
        company_col_2c = find_company_column(df_2c)
        if company_col_2c and "RECEIVED" in df_2c.columns:
            merged = merged.merge(df_2c[[company_col_2c, "RECEIVED"]], left_on="COMPANY", right_on=company_col_2c, how="left")
            metrics["total"] = merged["RECEIVED"].fillna(0).astype(int).values
            merged.drop(columns=["RECEIVED", company_col_2c], inplace=True, errors="ignore")

    # Unresolved (from 2A)
    if df_2a is not None:
        company_col_2a = find_company_column(df_2a)
        unresolved = df_2a[df_2a["STATUS"].str.upper() != "RESOLVED"]
        unresolved_counts = unresolved.groupby(company_col_2a).size().reset_index(name="UNRESOLVED")
        merged = merged.merge(unresolved_counts, left_on="COMPANY", right_on=company_col_2a, how="left")
        metrics["unresolved"] = merged["UNRESOLVED"].fillna(0).astype(int).values
        merged.drop(columns=["UNRESOLVED", company_col_2a], inplace=True, errors="ignore")

    # Recent (from 2A)
    if df_2a is not None:
        company_col_2a = find_company_column(df_2a)
        date_col_2a = None
        for c in df_2a.columns:
            if "DATE OF RECIEPT" in c or "DATE OF RECEIPT" in c:
                date_col_2a = c
                break
        if company_col_2a and date_col_2a:
            today = pd.Timestamp.today()
            one_year_ago = today - pd.Timedelta(days=365)
            df_2a["DATE_RECPT_DT"] = pd.to_datetime(df_2a[date_col_2a], errors="coerce")
            recent = df_2a[df_2a["DATE_RECPT_DT"] >= one_year_ago]
            recent_counts = recent.groupby(company_col_2a).size().reset_index(name="RECENT")
            merged = merged.merge(recent_counts, left_on="COMPANY", right_on=company_col_2a, how="left")
            metrics["recent"] = merged["RECENT"].fillna(0).astype(int).values
            merged.drop(columns=["RECENT", company_col_2a], inplace=True, errors="ignore")

    # Pending (from 2C)
    if df_2c is not None and "PENDING FOR REDRESSAL WITH EXCHANGE" in df_2c.columns:
        company_col_2c = find_company_column(df_2c)
        merged = merged.merge(
            df_2c[[company_col_2c, "PENDING FOR REDRESSAL WITH EXCHANGE"]],
            left_on="COMPANY", right_on=company_col_2c, how="left")
        metrics["pending"] = merged["PENDING FOR REDRESSAL WITH EXCHANGE"].fillna(0).astype(int).values
        merged.drop(columns=["PENDING FOR REDRESSAL WITH EXCHANGE", company_col_2c], inplace=True, errors="ignore")

    # Shareholders (from 2C)
    if df_2c is not None and "NO. OF SHAREHOLDERS" in df_2c.columns:
        company_col_2c = find_company_column(df_2c)
        merged = merged.merge(df_2c[[company_col_2c, "NO. OF SHAREHOLDERS"]],
                              left_on="COMPANY", right_on=company_col_2c, how="left")
        metrics["shareholders"] = merged["NO. OF SHAREHOLDERS"].fillna(0).astype(int).values
        merged.drop(columns=["NO. OF SHAREHOLDERS", company_col_2c], inplace=True, errors="ignore")

    # --- Min-max normalization for all metrics
    for key in metrics:
        metrics[key] = minmax_scale(pd.Series(metrics[key]))

    # --- Scoring ---
    # ENFORCEMENT_SCORE = 100 * (1 - w1*total - w2*unresolved - w3*recent - w4*pending + w5*shareholders)
    merged["ENFORCEMENT_SCORE"] = 100 * (
        1
        - w_total * metrics["total"]
        - w_unresolved * metrics["unresolved"]
        - w_recent * metrics["recent"]
        - w_pending * metrics["pending"]
        + w_shareholders * metrics["shareholders"]
    )

    # Clamp to [0, 100]
    merged["ENFORCEMENT_SCORE"] = merged["ENFORCEMENT_SCORE"].clip(lower=0, upper=100)

    # Rank (highest is best)
    merged["RANK"] = merged["ENFORCEMENT_SCORE"].rank(ascending=False, method='min').astype(int)

    result_cols = ["COMPANY", "INDUSTRY", "SYMBOL", "SERIES", "ISIN", "ENFORCEMENT_SCORE", "RANK"]
    result_cols = [c for c in result_cols if c in merged.columns]
    result_df = merged[result_cols].sort_values("RANK")
    st.subheader("Accurate Enforcement Index Results")
    st.dataframe(result_df, use_container_width=True)

    # --- Download options ---
    st.header("4. Download Results")
    out_excel = io.BytesIO()
    result_df.to_excel(out_excel, index=False, engine="openpyxl")
    st.download_button(
        label="Download as Excel",
        data=out_excel.getvalue(),
        file_name="enforcement_index_accurate.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    out_csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as CSV",
        data=out_csv,
        file_name="enforcement_index_accurate.csv",
        mime="text/csv"
    )

    st.markdown("> **Tip:** All variables are min-max normalized, scores are always in [0, 100]. Adjust weights for your business logic.")

else:
    st.info("Please upload the constituents file and at least one enforcement file to proceed.")

st.markdown("---")
st.markdown("Made with :orange[Streamlit] | [GitHub Copilot]")
