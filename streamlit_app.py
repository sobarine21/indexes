import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Enforcement Index Generator", layout="wide")

st.title("Enforcement Index Generator")

st.markdown("""
This tool helps you generate an Enforcement Index for companies based on enforcement records.
- **Step 1:** Upload the *Constituents* Excel file (list of companies to track, with optional weights for each company).
- **Step 2:** Upload the *Enforcement* Excel file (with enforcement actions per company).
- **Step 3:** The app will compute an index, rank the companies, and allow you to download the results.
""")

# --- File Uploads ---
st.header("1. Upload Files")
col1, col2 = st.columns(2)

with col1:
    constituents_file = st.file_uploader(
        "Upload Constituents Excel",
        type=["xlsx", "xls"],
        key="constituents"
    )

with col2:
    enforcement_file = st.file_uploader(
        "Upload Enforcement Excel",
        type=["xlsx", "xls"],
        key="enforcement"
    )

# --- Data Preview & Processing ---
if constituents_file and enforcement_file:
    st.header("2. Data Preview")
    
    # Read constituents file
    df_const = pd.read_excel(constituents_file)
    if 'COMPANY' not in df_const.columns and 'Name' in df_const.columns:
        df_const.rename(columns={'Name': 'COMPANY'}, inplace=True)
    # Assume columns: COMPANY, WEIGHT (optional)
    if 'WEIGHT' not in df_const.columns:
        df_const['WEIGHT'] = 1.0
    st.subheader("Constituents")
    st.dataframe(df_const)
    
    # Read enforcement file
    df_enf = pd.read_excel(enforcement_file)
    st.subheader("Enforcement Records")
    st.dataframe(df_enf.head(20))
    
    # --- Enforcement Index Calculation ---
    st.header("3. Enforcement Index Calculation")
    
    # Normalize company names for join
    df_const['COMPANY'] = df_const['COMPANY'].astype(str).str.upper().str.strip()
    df_enf['NAME OF COMPANY'] = df_enf['NAME OF COMPANY'].astype(str).str.upper().str.strip()
    
    # Compute enforcement counts per company
    enforcement_counts = (
        df_enf.groupby('NAME OF COMPANY')
        .size()
        .reset_index(name='ENFORCEMENT_COUNT')
    )
    
    # Merge with constituents
    index_df = pd.merge(
        df_const,
        enforcement_counts,
        left_on='COMPANY',
        right_on='NAME OF COMPANY',
        how='left'
    )
    index_df['ENFORCEMENT_COUNT'].fillna(0, inplace=True)
    index_df['ENFORCEMENT_COUNT'] = index_df['ENFORCEMENT_COUNT'].astype(int)
    
    # Example: Compute Index as inverse of enforcement counts weighted
    max_count = index_df['ENFORCEMENT_COUNT'].max()
    if max_count == 0:
        index_df['ENFORCEMENT_SCORE'] = 1.0
    else:
        index_df['ENFORCEMENT_SCORE'] = (1 - (index_df['ENFORCEMENT_COUNT'] / max_count)) * index_df['WEIGHT']
    
    # Ranking
    index_df['RANK'] = index_df['ENFORCEMENT_SCORE'].rank(ascending=False, method='min').astype(int)
    
    # Final columns
    output_df = index_df[[
        'COMPANY', 'WEIGHT', 'ENFORCEMENT_COUNT', 'ENFORCEMENT_SCORE', 'RANK'
    ]].sort_values('RANK')
    
    st.subheader("Enforcement Index Results")
    st.dataframe(output_df, use_container_width=True)
    
    # Download button
    st.header("4. Download Results")
    out_xlsx = output_df.to_excel(index=False, engine='openpyxl')
    st.download_button(
        label="Download Enforcement Index as Excel",
        data=out_xlsx,
        file_name="enforcement_index.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    
    st.markdown("**Note**: You can sort by any column in the preview above.")

else:
    st.info("Please upload both the constituents and enforcement files to proceed.")

st.markdown("---")
st.markdown("Made with :orange[Streamlit] | [GitHub Copilot]")
