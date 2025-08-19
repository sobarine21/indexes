import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="Enforcement Index Generator", layout="wide")

st.title("Enforcement Index Generator")

st.markdown("""
This tool generates an Enforcement Index for companies based on enforcement action records.

**How it works:**
- **Step 1:** Upload the *Constituents* file (Excel or CSV).
- **Step 2:** Upload the *Enforcement* file (Excel or CSV).
- **Step 3:** Customize index calculation and ranking options.
- **Step 4:** Download the computed index as Excel or CSV.
""")

# --- File Uploads ---
st.header("1. Upload Files")
col1, col2 = st.columns(2)

constituents_file = col1.file_uploader(
    "Upload Constituents File (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="constituents"
)
enforcement_file = col2.file_uploader(
    "Upload Enforcement File (Excel or CSV)",
    type=["xlsx", "xls", "csv"],
    key="enforcement"
)

# --- Load Data ---
def load_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

df_const, df_enf = None, None

if constituents_file:
    df_const = load_file(constituents_file)
    # Standardize column names
    df_const.columns = [c.strip().upper() for c in df_const.columns]
    # Rename to expected
    df_const.rename(columns={
        "COMPANY NAME": "COMPANY",
        "SYMBOL": "SYMBOL",
        "SERIES": "SERIES",
        "ISIN CODE": "ISIN",
        "INDUSTRY": "INDUSTRY"
    }, inplace=True)
    df_const['COMPANY'] = df_const['COMPANY'].astype(str).str.upper().str.strip()

if enforcement_file:
    df_enf = load_file(enforcement_file)
    df_enf.columns = [c.strip().upper() for c in df_enf.columns]
    # Try to standardize status column (make it always there, even if missing)
    possible_status_cols = ["STATUS", "COMPLAINT STATUS", "ENFORCEMENT STATUS"]
    status_col = None
    for col in possible_status_cols:
        if col in df_enf.columns:
            status_col = col
            break
    if status_col is None:
        # Add a dummy STATUS column with all "RESOLVED" so unresolved counts always zero
        df_enf["STATUS"] = "RESOLVED"
        status_col = "STATUS"
    if "NAME OF COMPANY" in df_enf.columns:
        df_enf['NAME OF COMPANY'] = df_enf['NAME OF COMPANY'].astype(str).str.upper().str.strip()

# --- Data Preview ---
if df_const is not None and df_enf is not None:
    st.header("2. Data Preview")

    st.subheader("Constituents")
    st.dataframe(df_const.head(20), use_container_width=True)

    st.subheader("Enforcement Records")
    st.dataframe(df_enf.head(20), use_container_width=True)

    st.header("3. Index Calculation Settings")

    # --- Customization Options ---
    st.markdown("#### Advanced Calculation Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        count_weight = st.number_input(
            "Weight for Number of Enforcements", min_value=0.0, max_value=10.0, value=1.0)
    with col2:
        recentness_weight = st.number_input(
            "Weight for Recent Enforcements (last 1 year)", min_value=0.0, max_value=10.0, value=2.0)
    with col3:
        unresolved_weight = st.number_input(
            "Weight for Unresolved Complaints", min_value=0.0, max_value=10.0, value=3.0)

    index_formula = st.text_area(
        "Custom Index Formula (use variables: total, recent, unresolved, count_wt, rec_wt, unres_wt)",
        value="1 / (1 + count_wt*total + rec_wt*recent + unres_wt*unresolved)",
        help="Define your own formula using available variables."
    )

    # --- Calculation ---
    st.header("4. Enforcement Index Calculation")

    # Handle date parsing for recent enforcements
    date_col = None
    for c in ["DATE OF RECIEPT", "DATE OF RECEIPT", "RECEIPT DATE", "DATE"]:
        if c in df_enf.columns:
            date_col = c
            break

    if date_col is not None:
        df_enf[date_col] = pd.to_datetime(df_enf[date_col], errors='coerce')
        today = pd.Timestamp.today()
        one_year_ago = today - pd.Timedelta(days=365)
    else:
        today = pd.Timestamp.today()
        one_year_ago = today - pd.Timedelta(days=365)
        df_enf["DUMMY_DATE"] = today
        date_col = "DUMMY_DATE"

    # Total enforcements
    if "NAME OF COMPANY" in df_enf.columns:
        group_col = "NAME OF COMPANY"
    elif "COMPANY" in df_enf.columns:
        group_col = "COMPANY"
    else:
        group_col = df_enf.columns[0]  # fallback to first column

    total_counts = df_enf.groupby(group_col).size().reset_index(name="TOTAL_ENFORCEMENTS")
    # Recent enforcements
    recent_counts = df_enf[df_enf[date_col] >= one_year_ago].groupby(group_col).size().reset_index(name="RECENT_ENFORCEMENTS")
    # Unresolved complaints
    unresolved_counts = df_enf[df_enf[status_col].str.upper() != "RESOLVED"].groupby(group_col).size().reset_index(name="UNRESOLVED")

    # Merge all counts
    score_df = df_const.copy()
    score_df = score_df.merge(total_counts, left_on="COMPANY", right_on=group_col, how="left")
    score_df = score_df.merge(recent_counts, left_on="COMPANY", right_on=group_col, how="left")
    score_df = score_df.merge(unresolved_counts, left_on="COMPANY", right_on=group_col, how="left")
    for col in ["TOTAL_ENFORCEMENTS", "RECENT_ENFORCEMENTS", "UNRESOLVED"]:
        if col not in score_df.columns:
            score_df[col] = 0
        score_df[col] = score_df[col].fillna(0).astype(int)

    # Prepare variables for formula
    count_wt = count_weight
    rec_wt = recentness_weight
    unres_wt = unresolved_weight

    # Calculate index using custom formula
    def safe_eval(formula, total, recent, unresolved, count_wt, rec_wt, unres_wt):
        # only allow numeric values and the specified variables
        allowed_names = {
            "total": total,
            "recent": recent,
            "unresolved": unresolved,
            "count_wt": count_wt,
            "rec_wt": rec_wt,
            "unres_wt": unres_wt,
            "np": np
        }
        try:
            return eval(formula, {"__builtins__": {}}, allowed_names)
        except Exception:
            return np.nan

    score_df["ENFORCEMENT_SCORE"] = score_df.apply(
        lambda row: safe_eval(
            index_formula,
            row["TOTAL_ENFORCEMENTS"],
            row["RECENT_ENFORCEMENTS"],
            row["UNRESOLVED"],
            count_wt,
            rec_wt,
            unres_wt
        ),
        axis=1
    )

    score_df["RANK"] = score_df["ENFORCEMENT_SCORE"].rank(ascending=False, method='min').astype(int)

    # Output columns
    show_cols = [
        "COMPANY", "INDUSTRY", "SYMBOL", "SERIES", "ISIN",
        "TOTAL_ENFORCEMENTS", "RECENT_ENFORCEMENTS", "UNRESOLVED",
        "ENFORCEMENT_SCORE", "RANK"
    ]
    output_cols = [c for c in show_cols if c in score_df.columns]
    output_df = score_df[output_cols].sort_values("RANK")

    st.subheader("Enforcement Index Results")
    st.dataframe(output_df, use_container_width=True)

    # --- Download options ---
    st.header("5. Download Results")
    out_excel = io.BytesIO()
    output_df.to_excel(out_excel, index=False, engine="openpyxl")
    st.download_button(
        label="Download as Excel",
        data=out_excel.getvalue(),
        file_name="enforcement_index.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    out_csv = output_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download as CSV",
        data=out_csv,
        file_name="enforcement_index.csv",
        mime="text/csv"
    )

    st.markdown("**Tip:** Adjust weights and formula above to experiment with ranking logic.")

else:
    st.info("Please upload both the constituents and enforcement files to proceed.")

st.markdown("---")
st.markdown("Made with :orange[Streamlit] | [GitHub Copilot]")
