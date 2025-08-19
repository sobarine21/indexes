import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Unified Enforcement Index Generator", layout="wide")
st.title("Unified Enforcement Index Generator")

st.markdown("""
Upload your *Constituents* file and up to two Enforcement files:
- **File 1:** Complaints/events (each row = one complaint)
- **File 2:** Company summary (each row = one company, summary stats)

The system will merge, preview, and let you customize the Enforcement Index calculation.
**All variables will be factored into calculations as per your formula and weights, but only result columns, not raw metric columns, will be shown in the final output.**
""")

# --- Helper Functions ---

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

def safe_eval(formula, **kwargs):
    allowed_names = {k: kwargs.get(k, 0) for k in [
        "total", "unresolved", "recent", "pending", "shareholders",
        "count_wt", "unres_wt", "rec_wt", "pend_wt", "sh_wt", "np"
    ]}
    allowed_names["np"] = np
    try:
        return eval(formula, {"__builtins__": {}}, allowed_names)
    except Exception:
        return np.nan

# --- File Uploads ---
st.header("1. Upload Files")
col1, col2, col3 = st.columns(3)

constituents_file = col1.file_uploader(
    "Upload Constituents File (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="constituents"
)
enforcement_file_1 = col2.file_uploader(
    "Upload Enforcement File 1: Events (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="enforcement1"
)
enforcement_file_2 = col3.file_uploader(
    "Upload Enforcement File 2: Summary (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="enforcement2"
)

df_const, df_enf1, df_enf2 = None, None, None

# --- Load and Standardize Data ---
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

if enforcement_file_1:
    df_enf1 = load_file(enforcement_file_1)
    df_enf1 = clean_and_standardize(df_enf1)
    company_col1 = find_company_column(df_enf1)
    if company_col1:
        df_enf1[company_col1] = df_enf1[company_col1].astype(str).str.upper().str.strip()
    if "STATUS" not in df_enf1.columns:
        df_enf1["STATUS"] = "RESOLVED"
    # Parse date if available
    date_col1 = None
    for c in df_enf1.columns:
        if "DATE OF RECIEPT" in c or "DATE OF RECEIPT" in c:
            date_col1 = c
            break
    if date_col1 is None:
        df_enf1["DATE OF RECIEPT"] = pd.NaT
        date_col1 = "DATE OF RECIEPT"
    df_enf1[date_col1] = pd.to_datetime(df_enf1[date_col1], errors="coerce")

if enforcement_file_2:
    df_enf2 = load_file(enforcement_file_2)
    df_enf2 = clean_and_standardize(df_enf2)
    company_col2 = find_company_column(df_enf2)
    if company_col2:
        df_enf2[company_col2] = df_enf2[company_col2].astype(str).str.upper().str.strip()
    num_cols = [
        "NO. OF SHAREHOLDERS", "RECEIVED", "REDRESSED THROUGH EXCHANGE",
        "NON-ACTIONABLE~", "ADVISED / OPTED FOR ARBITRATION", "PENDING FOR REDRESSAL WITH EXCHANGE"
    ]
    for c in num_cols:
        to_numeric_safe(df_enf2, c)
    # Drop all date columns in summary except those used in calc
    for c in list(df_enf2.columns):
        if 'DATE' in c and c not in num_cols:
            df_enf2.drop(columns=[c], inplace=True, errors="ignore")

# --- Data Preview ---
if df_const is not None:
    st.subheader("Constituents Preview")
    st.dataframe(df_const.head(20), use_container_width=True)
if df_enf1 is not None:
    st.subheader("Enforcement File 1 (Events) Preview")
    st.dataframe(df_enf1.head(20), use_container_width=True)
if df_enf2 is not None:
    st.subheader("Enforcement File 2 (Summary) Preview")
    st.dataframe(df_enf2.head(20), use_container_width=True)

# --- Calculation Options ---
if df_const is not None and (df_enf1 is not None or df_enf2 is not None):
    st.header("2. Calculation Customization")

    # Let user select which data source for each metric
    metrics = [
        ("Total Complaints", "total_complaints"),
        ("Unresolved Complaints", "unresolved"),
        ("Recent Complaints (last 1yr)", "recent"),
        ("Pending for Redressal", "pending"),
        ("Shareholders", "shareholders")
    ]
    metric_sources = {}
    st.markdown("#### Select data source for each metric:")
    metric_cols = st.columns(len(metrics))
    for idx, (label, var) in enumerate(metrics):
        choices = []
        if df_enf1 is not None and var in ["total_complaints","unresolved","recent"]:
            choices.append("File 1")
        if df_enf2 is not None and var in ["total_complaints","pending","shareholders"]:
            choices.append("File 2")
        if choices:
            default = 0
            metric_sources[var] = metric_cols[idx].selectbox(label, choices, index=default, key=f"src_{var}")
        else:
            metric_sources[var] = None

    wcols = st.columns(5)
    count_weight = wcols[0].number_input("Weight: Total Complaints", 0.0, 10.0, 1.0)
    unresolved_weight = wcols[1].number_input("Weight: Unresolved", 0.0, 10.0, 2.0)
    recent_weight = wcols[2].number_input("Weight: Recent (1yr)", 0.0, 10.0, 2.0)
    pending_weight = wcols[3].number_input("Weight: Pending", 0.0, 10.0, 2.0)
    shareholders_weight = wcols[4].number_input("Weight: Shareholders", 0.0, 10.0, 0.0)

    custom_formula = st.text_area(
        "Custom Index Formula (use: total, unresolved, recent, pending, shareholders, count_wt, unres_wt, rec_wt, pend_wt, sh_wt)",
        value="1/(1 + count_wt*total + unres_wt*unresolved + rec_wt*recent + pend_wt*pending - sh_wt*shareholders/10000)",
        help="Higher score = better. You can set shareholders_weight=0 to ignore that variable."
    )

    # --- Metric Calculation ---
    st.header("3. Calculation and Ranking")
    merged = df_const.copy()

    # Calculate all metrics, but do NOT show raw columns in output (only use for scoring)
    metric_values = {}

    # -- Total Complaints
    metric_values["total"] = np.zeros(len(merged))
    if metric_sources["total_complaints"] == "File 1" and df_enf1 is not None:
        company_col1 = find_company_column(df_enf1)
        if company_col1:
            total_counts = df_enf1.groupby(company_col1).size().reset_index(name="TOTAL_COMPLAINTS_F1")
            merged = merged.merge(total_counts, left_on="COMPANY", right_on=company_col1, how="left")
            metric_values["total"] = merged["TOTAL_COMPLAINTS_F1"].fillna(0).astype(int).values
            merged.drop(columns=["TOTAL_COMPLAINTS_F1", company_col1], inplace=True, errors="ignore")
    elif metric_sources["total_complaints"] == "File 2" and df_enf2 is not None:
        company_col2 = find_company_column(df_enf2)
        if company_col2 and "RECEIVED" in df_enf2.columns:
            merged = merged.merge(df_enf2[[company_col2, "RECEIVED"]], left_on="COMPANY", right_on=company_col2, how="left")
            metric_values["total"] = merged["RECEIVED"].fillna(0).astype(int).values
            merged.drop(columns=["RECEIVED", company_col2], inplace=True, errors="ignore")

    # -- Unresolved
    metric_values["unresolved"] = np.zeros(len(merged))
    if metric_sources["unresolved"] == "File 1" and df_enf1 is not None:
        company_col1 = find_company_column(df_enf1)
        if company_col1:
            unresolved = df_enf1[df_enf1["STATUS"].str.upper() != "RESOLVED"]
            unresolved_counts = unresolved.groupby(company_col1).size().reset_index(name="UNRESOLVED_F1")
            merged = merged.merge(unresolved_counts, left_on="COMPANY", right_on=company_col1, how="left")
            metric_values["unresolved"] = merged["UNRESOLVED_F1"].fillna(0).astype(int).values
            merged.drop(columns=["UNRESOLVED_F1", company_col1], inplace=True, errors="ignore")

    # -- Recent
    metric_values["recent"] = np.zeros(len(merged))
    if metric_sources["recent"] == "File 1" and df_enf1 is not None:
        company_col1 = find_company_column(df_enf1)
        date_col1 = None
        for c in df_enf1.columns:
            if "DATE OF RECIEPT" in c or "DATE OF RECEIPT" in c:
                date_col1 = c
                break
        if company_col1 and date_col1:
            today = pd.Timestamp.today()
            one_year_ago = today - pd.Timedelta(days=365)
            df_enf1["DATE_RECPT_DT"] = pd.to_datetime(df_enf1[date_col1], errors="coerce")
            recent = df_enf1[df_enf1["DATE_RECPT_DT"] >= one_year_ago]
            recent_counts = recent.groupby(company_col1).size().reset_index(name="RECENT_F1")
            merged = merged.merge(recent_counts, left_on="COMPANY", right_on=company_col1, how="left")
            metric_values["recent"] = merged["RECENT_F1"].fillna(0).astype(int).values
            merged.drop(columns=["RECENT_F1", company_col1], inplace=True, errors="ignore")

    # -- Pending
    metric_values["pending"] = np.zeros(len(merged))
    if metric_sources["pending"] == "File 2" and df_enf2 is not None:
        company_col2 = find_company_column(df_enf2)
        if company_col2 and "PENDING FOR REDRESSAL WITH EXCHANGE" in df_enf2.columns:
            merged = merged.merge(
                df_enf2[[company_col2, "PENDING FOR REDRESSAL WITH EXCHANGE"]],
                left_on="COMPANY", right_on=company_col2, how="left")
            metric_values["pending"] = merged["PENDING FOR REDRESSAL WITH EXCHANGE"].fillna(0).astype(int).values
            merged.drop(columns=["PENDING FOR REDRESSAL WITH EXCHANGE", company_col2], inplace=True, errors="ignore")

    # -- Shareholders
    metric_values["shareholders"] = np.zeros(len(merged))
    if metric_sources["shareholders"] == "File 2" and df_enf2 is not None:
        company_col2 = find_company_column(df_enf2)
        if company_col2 and "NO. OF SHAREHOLDERS" in df_enf2.columns:
            merged = merged.merge(df_enf2[[company_col2, "NO. OF SHAREHOLDERS"]],
                                  left_on="COMPANY", right_on=company_col2, how="left")
            metric_values["shareholders"] = merged["NO. OF SHAREHOLDERS"].fillna(0).astype(int).values
            merged.drop(columns=["NO. OF SHAREHOLDERS", company_col2], inplace=True, errors="ignore")

    # --- Compute Index and Rank ---
    merged["ENFORCEMENT_SCORE"] = [
        safe_eval(
            custom_formula,
            total=int(metric_values["total"][i]),
            unresolved=int(metric_values["unresolved"][i]),
            recent=int(metric_values["recent"][i]),
            pending=int(metric_values["pending"][i]),
            shareholders=int(metric_values["shareholders"][i]),
            count_wt=count_weight,
            unres_wt=unresolved_weight,
            rec_wt=recent_weight,
            pend_wt=pending_weight,
            sh_wt=shareholders_weight
        )
        for i in range(len(merged))
    ]
    merged["RANK"] = merged["ENFORCEMENT_SCORE"].rank(ascending=False, method='min').astype(int)

    # --- Output: Only show result columns, not raw metrics
    result_cols = ["COMPANY", "INDUSTRY", "SYMBOL", "SERIES", "ISIN", "ENFORCEMENT_SCORE", "RANK"]
    result_cols = [c for c in result_cols if c in merged.columns]
    result_df = merged[result_cols].sort_values("RANK")
    st.subheader("Unified Enforcement Index Results")
    st.dataframe(result_df, use_container_width=True)

    # --- Download options ---
    st.header("4. Download Results")
    out_excel = io.BytesIO()
    result_df.to_excel(out_excel, index=False, engine="openpyxl")
    st.download_button(
        label="Download as Excel",
        data=out_excel.getvalue(),
        file_name="enforcement_index_unified.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    out_csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as CSV",
        data=out_csv,
        file_name="enforcement_index_unified.csv",
        mime="text/csv"
    )

    st.markdown("> **Tip:** All selected variables are used in scoring, but only result columns are visible in output.")

else:
    st.info("Please upload the constituents file and at least one enforcement file to proceed.")

st.markdown("---")
st.markdown("Made with :orange[Streamlit] | [GitHub Copilot]")
