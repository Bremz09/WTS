import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="MTP Drafting Overview", layout="wide")
st.title("MTP Drafting Overview")

uploaded_file = st.file_uploader("Upload rep Excel file", type=["xlsx"])

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file, sheet_name="Overview", header=None)

    # Get segment numbers from the overview 'Segment' row (col B = index 1)
    seg_row = df_raw[df_raw[1] == "Segment"]
    if not seg_row.empty:
        segment_cols = [c for c in seg_row.iloc[0, 2:].tolist() if pd.notna(c)]
    else:
        segment_cols = list(range(1, df_raw.shape[1] - 1))

    # Athlete blocks start where col A is not null and not 'Overview'
    athlete_idx = df_raw.index[df_raw[0].notna() & (df_raw[0] != "Overview")].tolist()

    if not athlete_idx:
        st.warning("No athlete data found on the Overview sheet.")
    else:
        athletes = []
        boundaries = athlete_idx + [len(df_raw)]
        for i, start in enumerate(athlete_idx):
            end = boundaries[i + 1]
            block = df_raw.iloc[start:end]
            name = block.iloc[0, 0].split(" - ")[0]
            block = block[block[1].notna()]
            metrics = block[1].tolist()
            values = block.iloc[:, 2: 2 + len(segment_cols)].values
            n_segs = values.shape[1]
            cols = segment_cols[:n_segs]
            athlete_df = pd.DataFrame(values, index=metrics, columns=cols)
            athlete_df = athlete_df.where(athlete_df.notna(), other=None)
            athlete_df.index.name = "Metric"
            athlete_df.columns.name = "Segment"
            athletes.append((name, athlete_df))

        for name, adf in athletes:
            st.subheader(name)
            st.dataframe(adf.T, use_container_width=True)

        # Lead rider stats per segment
        lead_rows = []
        for seg in segment_cols:
            for name, adf in athletes:
                if seg in adf.columns and adf.loc["Position", seg] == 1:
                    lead_rows.append({
                        "Segment": seg,
                        "Lead Rider": name,
                        "Power": adf.loc["Power", seg] if "Power" in adf.index else None,
                        "CdA": adf.loc["CdA", seg] if "CdA" in adf.index else None,
                        "Average Speed": adf.loc["Average Speed", seg] if "Average Speed" in adf.index else None,
                    })
                    break

        lead_df = pd.DataFrame(lead_rows)
        lead_df["W/CdA"] = lead_df.apply(
            lambda r: round(r["Power"] / r["CdA"], 2) if r["Power"] and r["CdA"] else None, axis=1
        )
        st.subheader("Lead Rider by Segment")
        st.dataframe(lead_df, use_container_width=True)

        plot_df = lead_df[["W/CdA", "Average Speed"]].dropna()
        if not plot_df.empty:
            x = plot_df["W/CdA"].values.astype(float)
            y = plot_df["Average Speed"].values.astype(float)
            m, b = np.polyfit(x, y, 1)
            y_pred = m * x + b
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

            fig = go.Figure()
            for rider, group in plot_df.join(lead_df["Lead Rider"]).groupby("Lead Rider"):
                rx = group["W/CdA"].values.astype(float)
                ry = group["Average Speed"].values.astype(float)
                fig.add_trace(go.Scatter(
                    x=rx, y=ry,
                    mode="markers",
                    name=rider,
                    marker=dict(size=8),
                ))
            x_line = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(go.Scatter(
                x=x_line, y=m * x_line + b,
                mode="lines",
                name=f"Trendline (R²={r2:.4f})",
                line=dict(dash="dash"),
            ))
            fig.update_layout(
                title=f"Average Speed vs W/CdA — R² = {r2:.4f}",
                xaxis_title="W/CdA",
                yaxis_title="Average Speed (km/h)",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Wobble vs Distance for all riders and segments
        wobble_rows = []
        for name, adf in athletes:
            for seg in segment_cols:
                if seg in adf.columns:
                    wobble_rows.append({
                        "Rider": name,
                        "Segment": seg,
                        "Wobble": adf.loc["Wobble", seg] if "Wobble" in adf.index else None,
                        "Distance": adf.loc["Distance", seg] if "Distance" in adf.index else None,
                    })

        wobble_df = pd.DataFrame(wobble_rows)
        st.subheader("Wobble & Distance — All Riders")
        st.dataframe(wobble_df, use_container_width=True)

        plot_w = wobble_df[["Rider", "Wobble", "Distance"]].dropna()
        if not plot_w.empty:
            fig2 = go.Figure()
            for rider, group in plot_w.groupby("Rider"):
                fig2.add_trace(go.Scatter(
                    x=group["Distance"].values.astype(float),
                    y=group["Wobble"].values.astype(float),
                    mode="markers",
                    name=rider,
                    marker=dict(size=8),
                ))
            fig2.update_layout(
                title="Wobble vs Distance",
                xaxis_title="Distance (m)",
                yaxis_title="Wobble",
            )
            st.plotly_chart(fig2, use_container_width=True)
