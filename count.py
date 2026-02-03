import streamlit as st
import pandas as pd
import io
 
st.set_page_config(page_title="Attendance/Feature Counter", layout="wide")
st.title("✅ Present/Absent Counter (CSV / Excel)")
 
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
 
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # Try utf-8; if your file is from Windows, latin-1 may be needed
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    else:
        return pd.read_excel(uploaded_file, engine="openpyxl")
 
if uploaded is not None:
    df = load_file(uploaded)
    st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
 
    with st.expander("Preview"):
        st.dataframe(df.head(25), use_container_width=True)
 
    st.subheader("1) Choose feature/attendance columns")
    auto = st.checkbox("Auto-detect feature columns (recommended)", value=True)
 
    if auto:
        # Auto-detect columns that look like Present/Absent / P/A / Yes/No / 0/1
        candidates = []
        present_set = {"present", "p", "yes", "y", "true", "1"}
        absent_set  = {"absent", "a", "no", "n", "false", "0", ""}
 
        for c in df.columns:
            # consider only non-null uniques, normalize
            uniques = (
                df[c]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )
            # Skip empty columns
            if len(uniques) == 0:
                continue
            # If most values are in present/absent sets, treat as a feature column
            if all(u in (present_set | absent_set) for u in uniques) and len(uniques) <= 6:
                candidates.append(c)
 
        feature_cols = st.multiselect(
            "Auto-detected columns (edit if needed)",
            options=df.columns.tolist(),
            default=candidates
        )
    else:
        feature_cols = st.multiselect(
            "Select columns to count",
            options=df.columns.tolist(),
            default=[]
        )
 
    st.subheader("2) Define which values count as PRESENT")
    present_input = st.text_input(
        "Present values (comma-separated, case-insensitive)",
        value="present, p, yes, y, true, 1"
    )
    present_values = {v.strip().lower() for v in present_input.split(",") if v.strip()}
 
    if feature_cols:
        norm = df[feature_cols].astype(str).apply(lambda s: s.str.strip().str.lower())
        present_mask = norm.isin(present_values)
 
        df["present_count"] = present_mask.sum(axis=1)
        df["absent_count"] = (~present_mask).sum(axis=1)
 
        st.subheader("✅ Output")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Avg present/row", round(df["present_count"].mean(), 2))
        c3.metric("Max present in a row", int(df["present_count"].max()))
 
        st.dataframe(df, use_container_width=True)
 
        st.subheader("Feature-wise Present totals")
        totals = present_mask.sum(axis=0).sort_values(ascending=False).reset_index()
        totals.columns = ["feature", "present_total"]
        st.dataframe(totals, use_container_width=True)
 
        # Downloads
        st.subheader("⬇️ Download results")
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="attendance_with_counts.csv",
            mime="text/csv"
        )
 
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="results")
            totals.to_excel(writer, index=False, sheet_name="feature_totals")
        st.download_button(
            "Download Excel",
            buffer.getvalue(),
            file_name="attendance_with_counts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("Select at least one feature column.")