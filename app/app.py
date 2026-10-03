"""Thai Seafood Export Outlook — แอป Streamlit สำหรับผลลัพธ์โครงงาน ML (main.ipynb)

รัน (จากโฟลเดอร์โปรเจกต์):  streamlit run app/app.py
ต้องมี:  pip install -r app/requirements.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

DATA = core.DATA_DIR
st.set_page_config(page_title="Thai Seafood Export Outlook", page_icon="🦐", layout="wide")

GROUP_NAMES = {"A": "A ตลาดหลัก: ใหญ่ ต่อเนื่อง ผันผวนต่ำ", "B": "B ตลาดปลากระป๋องราคาต่ำที่หดตัว", "C": "C ตลาดโตเร็ว ราคาต่อกก. สูง", "D": "D ตลาดเล็ก ผันผวน ค้าไม่ต่อเนื่อง", "นอกกลุ่ม": "นอกกลุ่ม (ข้อมูลไม่พอจัดกลุ่ม)"}
GROUP_COLORS = {"A": "#2563eb", "B": "#f97316", "C": "#16a34a", "D": "#7c3aed", "นอกกลุ่ม": "#6b7280"}
RELIABILITY = {"A": ("🟢", "เชื่อถือได้พอใช้ — ใช้ประกอบการวางแผนได้ (พร้อมช่วง 80%)"), "B": ("🟠", "ความผิดพลาดสูง — ใช้เป็นภาพรวม ไม่ใช้ตัดสินใจรายตัว"),
               "C": ("🔴", "ความผิดพลาดสูงมาก — ไม่ควรใช้ตัวเลขรายตัวตัดสินใจ"), "D": ("🔴", "ความผิดพลาดสูงมาก — ไม่ควรใช้ตัวเลขรายตัวตัดสินใจ"), "นอกกลุ่ม": ("🔴", "ข้อมูลไม่พอประเมินความน่าเชื่อถือ")}


# ------------------------------------------------------------------------------------------------ ข้อมูล
def rd(path, **kw):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], encoding="utf-8-sig", **kw)


@st.cache_data(show_spinner=False)
def load_data():
    d = {}
    d["hist"] = rd(DATA / "history.csv", dtype={"hs4": str}, parse_dates=["date"])
    d["fwd"] = rd(DATA / "forecast_latest.csv", dtype={"hs4": str}, parse_dates=["origin", "target_month"])
    d["bt"] = rd(DATA / "forecast_test_predictions.csv", dtype={"hs4": str}, parse_dates=["origin", "target_month"])
    d["ew"] = rd(DATA / "ew_latest.csv", dtype={"hs4": str})
    d["stat"] = rd(DATA / "series_static.csv", dtype={"hs4": str})
    d["clu"] = rd(DATA / "clusters_w2.csv")
    d["res"] = rd(DATA / "results_summary.csv")
    c = rd(DATA / "countries.csv"); d["names"] = dict(zip(c.iso2, c.name_th.fillna(c.name_en)))
    d["names_en"] = dict(zip(c.iso2, c.name_en))
    d["group"] = {r.iso2: r.cluster_name.split()[0] for r in d["clu"].itertuples()}
    return d


@st.cache_resource(show_spinner=False)
def get_models():
    return core.load_models(), core.load_meta()


D = load_data()
MODELS, META = get_models()
LAST = pd.Timestamp(META["last_month"])


def cname(iso2): return f"{D['names'].get(iso2, D['names_en'].get(iso2, iso2))} ({iso2})"
def group_of(iso2): return D["group"].get(iso2, "นอกกลุ่ม")
def hs_label(hs4): return f"{hs4} {core.HS4_NAME[hs4]}"
def fmt_m(v): return f"{v / 1e6:,.1f}" if np.isfinite(v) else "—"
def res(goal, model, setting, metric):
    t = D["res"]; r = t[(t["โจทย์"] == goal) & (t["โมเดล/วิธี"] == model) & (t["ตั้งค่า"] == setting) & (t["ตัวชี้วัด"] == metric)]
    return float(r["ค่า"].iloc[0]) if len(r) else np.nan


def series_values(hs4, iso2):
    return D["hist"][(D["hist"].hs4 == hs4) & (D["hist"].iso2 == iso2)].sort_values("date").value_usd.to_numpy()


def static_of(hs4, iso2):
    return D["stat"][(D["stat"].hs4 == hs4) & (D["stat"].iso2 == iso2)].iloc[0].to_dict()


def forecast_series(x, static):
    ff = core.hist_features(x, static)
    return ff, core.predict_forecast(MODELS, META, [ff]).set_index("h")


def series_picker(key, default_iso="US", default_hs="1604"):
    isos = sorted(D["stat"].iso2.unique(), key=lambda i: cname(i))
    iso = st.selectbox("ประเทศ/ตลาด", isos, index=isos.index(default_iso) if default_iso in isos else 0, format_func=cname, key=f"{key}_c")
    hss = sorted(D["stat"][D["stat"].iso2 == iso].hs4.unique())
    hs = st.selectbox("กลุ่มสินค้า (HS4)", hss, index=hss.index(default_hs) if default_hs in hss else 0, format_func=hs_label, key=f"{key}_h")
    return iso, hs


def months_between(a, b): return (b.year - a.year) * 12 + b.month - a.month


# ------------------------------------------------------------------------------------------------ เมนู
st.sidebar.title("🦐 Thai Seafood Export Outlook")
st.sidebar.caption("ผลลัพธ์โครงงาน ML: พยากรณ์ · จัดกลุ่มตลาด · เตือนตลาดยอดตก")
PAGE = st.sidebar.radio("เมนู", ["🏠 ภาพรวม", "🗺️ กลุ่มตลาด", "📈 พยากรณ์", "🚨 Early Warning", "🧪 สถานการณ์จำลอง", "ℹ️ วิธีอ่านและข้อจำกัด"], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown(f"**ข้อมูลถึง:** {LAST:%b %Y}  \n**แหล่งข้อมูล:** UN Comtrade (ส่งออกของไทย, USD)  \n**ขอบเขต:** {META['n_series']} อนุกรม (ประเทศ × กลุ่ม HS4) ราว 97% ของมูลค่า")
st.sidebar.warning("ผลนี้เป็นค่าพยากรณ์จากโมเดลสถิติ/ML ไม่ใช่คำแนะนำทางการลงทุน ตลาดกลุ่ม B–D ผิดพลาดสูง")

# ================================================================================================ 1) ภาพรวม
if PAGE.startswith("🏠"):
    st.title("🦐 ภาพรวมการส่งออกสินค้าประมงของไทย")
    st.caption("พยากรณ์ล่วงหน้า · จัดกลุ่มตลาด · เตือนตลาดที่ยอดกำลังจะตก — จากข้อมูลเปิด UN Comtrade (HS 4 หลัก 10 กลุ่ม, ม.ค. 2010 – เม.ย. 2026)")
    w1 = res("Goal 1 พยากรณ์", "LightGBM", "ระยะ 1 เดือน · ระดับยอดรวม", "WAPE"); w6 = res("Goal 1 พยากรณ์", "LightGBM", "ระยะ 6 เดือน · ระดับยอดรวม", "WAPE")
    p10 = res("Goal 3 Early Warning", "LightGBM", "ทุกตลาด ทดสอบ", "Precision เตือน 10%")
    clu = D["clu"].assign(v=lambda d: np.exp(d.size_log)); shareA = clu[clu.cluster_name.str.startswith("A")].v.sum() / clu.v.sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ผิดพลาดพยากรณ์ยอดรวม 1 เดือน", f"{w1:.1%}", f"ที่ 6 เดือน {w6:.1%}", delta_color="off", delta_arrow="off")
    c2.metric("ตลาดหลัก (กลุ่ม A) ครองมูลค่า", f"{shareA:.0%}", "จาก 35 ประเทศ", delta_color="off", delta_arrow="off")
    c3.metric("Early Warning แม่นเมื่อเตือน 10%", f"{p10:.0%}", "lift ~4 เท่าของการสุ่ม", delta_color="off", delta_arrow="off")
    c4.metric("ข้อมูลที่ใช้ (อนุกรม)", f"{META['n_series']}", "196 เดือน", delta_color="off", delta_arrow="off")

    tot = D["hist"].groupby("date").value_usd.sum() / 1e6; f = D["fwd"].groupby(["h", "target_month"]).pred.sum().reset_index()
    fig = go.Figure()
    fig.add_scatter(x=tot.index[-60:], y=tot.values[-60:], mode="lines", name="ยอดส่งออกรวมจริง", line=dict(color="#111827", width=2.2))
    fig.add_scatter(x=f.target_month, y=f.pred / 1e6, mode="markers+lines", name="พยากรณ์ยอดรวม (1/3/6 เดือนข้างหน้า)", marker=dict(size=11, symbol="diamond", color="#2563eb"), line=dict(dash="dot", color="#2563eb"))
    fig.add_vline(x=LAST, line_dash="dash", line_color="#9ca3af"); fig.add_annotation(x=LAST, y=1, yref="paper", text="ปัจจุบัน", showarrow=False, yanchor="bottom", font=dict(color="#6b7280"))
    fig.update_layout(title="ยอดส่งออกสินค้าประมงรวมรายเดือน (ล้านดอลลาร์สหรัฐ) — ย้อนหลัง 5 ปี และพยากรณ์ล่วงหน้า", height=420, yaxis_title="ล้านดอลลาร์", template="plotly_white", legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.caption("พยากรณ์ยอดรวมคือผลรวมของค่าพยากรณ์ 332 อนุกรม (ช่วงความไม่แน่นอนใช้ได้เฉพาะรายอนุกรม จึงไม่แสดงช่วงรวม) ความผิดพลาดของยอดรวมที่ทดสอบย้อนหลัง 36 เดือนอยู่ที่ราว 6–8%")

    st.subheader("ผลสรุปของโครงงาน")
    st.markdown("""
| โจทย์ | ผลที่ได้ | ใช้ทำอะไรได้ |
|---|---|---|
| **พยากรณ์ยอดรายเดือน** | LightGBM ชนะ baseline ที่ 1 และ 3 เดือน (ต่ำกว่า SES ~5%) ที่ 6 เดือนเสมอกับวิธีง่าย ๆ | วางแผนยอดรวมและตลาดหลัก 1–3 เดือน พร้อมช่วง 80% |
| **จัดกลุ่มตลาด** | 4 กลุ่ม — ตลาดหลัก 35 ประเทศครอง 92% ของมูลค่า | กำหนดกลยุทธ์แยกกลุ่ม |
| **Early Warning** | เตือน 10% ของตลาดที่เสี่ยงสุดต่อเดือน ตกจริง ~89% | ตรวจสอบตลาดที่ยอดกำลังจะตก |
""")
    st.info("👈 เลือกหน้าจากเมนูด้านซ้าย: **พยากรณ์** (ดูรายตลาด + ช่วง 80%), **Early Warning** (ตลาดเสี่ยง), **สถานการณ์จำลอง** (ลองปรับยอดล่าสุดแล้วดูว่าโมเดลทำนายเปลี่ยนอย่างไร)")

# ================================================================================================ 2) กลุ่มตลาด
elif PAGE.startswith("🗺️"):
    st.title("🗺️ กลุ่มตลาดส่งออก")
    st.caption("จัดกลุ่มด้วย K-Means (k = 4) จากขนาดตลาด การเติบโต ความผันผวน ราคาต่อกก. ความต่อเนื่อง และส่วนผสมสินค้า (ช่วง พ.ค. 2023 – เม.ย. 2026, 119 ประเทศ)")
    clu = D["clu"].copy(); clu["กลุ่ม"] = clu.cluster_name.str.split().str[0]; clu["value_usd"] = np.exp(clu.size_log); clu["ประเทศ"] = [cname(i) for i in clu.iso2]; clu["usd_per_kg"] = np.exp(clu.uv_log)
    g = clu.groupby("cluster_name").agg(ประเทศ=("iso2", "size"), มูลค่า_ล้านUSD=("value_usd", lambda s: s.sum() / 1e6), เติบโต=("growth", "mean"), ผันผวน=("volatility", "median"), ราคาต่อกก=("usd_per_kg", "median")).reset_index()
    g["% ของมูลค่า"] = g.มูลค่า_ล้านUSD / g.มูลค่า_ล้านUSD.sum() * 100
    st.dataframe(g.rename(columns={"cluster_name": "กลุ่ม", "มูลค่า_ล้านUSD": "มูลค่า 36 เดือน (ล้าน USD)", "เติบโต": "การเติบโตเฉลี่ย (log)", "ผันผวน": "ความผันผวน (มัธยฐาน)", "ราคาต่อกก": "ราคา USD/กก. (มัธยฐาน)"}),
                 hide_index=True, width="stretch", column_config={"% ของมูลค่า": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100), "มูลค่า 36 เดือน (ล้าน USD)": st.column_config.NumberColumn(format="%.0f"),
                                                                           "การเติบโตเฉลี่ย (log)": st.column_config.NumberColumn(format="%.2f"), "ความผันผวน (มัธยฐาน)": st.column_config.NumberColumn(format="%.2f"), "ราคา USD/กก. (มัธยฐาน)": st.column_config.NumberColumn(format="%.2f")})
    fig = px.scatter(clu, x="value_usd", y="growth", color="กลุ่ม", hover_name="ประเทศ", color_discrete_map=GROUP_COLORS, log_x=True, size_max=14,
                     hover_data={"value_usd": ":,.0f", "growth": ":.2f", "volatility": ":.2f", "กลุ่ม": False}, labels={"value_usd": "มูลค่าส่งออก 36 เดือน (USD, สเกล log)", "growth": "การเติบโต (log อัตราส่วนยอด 12 เดือนล่าสุดต่อ 12 เดือนแรก)"},
                     category_orders={"กลุ่ม": ["A", "B", "C", "D"]})
    fig.update_traces(marker=dict(size=10, opacity=.8, line=dict(width=.5, color="white"))); fig.add_hline(y=0, line_dash="dot", line_color="#9ca3af")
    fig.update_layout(height=470, template="plotly_white", title="แผนที่ตลาด: ขนาด × การเติบโต (เลื่อนเมาส์ดูชื่อประเทศ)")
    st.plotly_chart(fig, width="stretch")
    st.caption("⚠️ ตลาดกระจายต่อเนื่อง ไม่ได้แยกเป็นกลุ่มธรรมชาติชัดเจน (Silhouette 0.22) ผลนี้เป็นการแบ่งส่วนเชิงธุรกิจ ไม่ใช่ขอบเขตที่แน่นอน")
    st.subheader("ค้นหาประเทศ")
    iso = st.selectbox("เลือกประเทศ", sorted(clu.iso2, key=cname), format_func=cname, index=sorted(clu.iso2, key=cname).index("US") if "US" in set(clu.iso2) else 0)
    r = clu[clu.iso2 == iso].iloc[0]; cA, cB = st.columns([1, 1])
    with cA:
        st.markdown(f"### {cname(iso)}\n**กลุ่ม:** {r.cluster_name}")
        st.write(f"- มูลค่าส่งออก 36 เดือน ≈ **{r.value_usd / 1e6:,.1f} ล้านดอลลาร์**\n- การเติบโต: {r.growth:+.2f} · ความผันผวน: {r.volatility:.2f}\n- ราคาเฉลี่ย ≈ {r.usd_per_kg:,.2f} USD/กก. · ค้าขายใน {r.active_share:.0%} ของเดือน")
        if pd.notna(r.region): st.write(f"- ภูมิภาค: {r.region} · ระดับรายได้: {r.income}")
    with cB:
        mix = pd.Series({hs: r[f"mix_{hs}"] for hs in core.HS4_NAME}).rename(index=lambda h: hs_label(h)); mix = mix[mix > 0.005].sort_values()
        figm = go.Figure(go.Bar(x=mix.values * 100, y=mix.index, orientation="h", marker_color="#2563eb")); figm.update_layout(height=300, template="plotly_white", title="ส่วนผสมสินค้าที่ส่งออก (% ของมูลค่า)", margin=dict(l=0, r=10, t=40, b=0))
        st.plotly_chart(figm, width="stretch")

# ================================================================================================ 3) พยากรณ์
elif PAGE.startswith("📈"):
    st.title("📈 พยากรณ์ยอดส่งออกรายตลาด")
    st.caption("โมเดล LightGBM เทรนรวมทุกอนุกรม · ย้อนหลัง = ผลทดสอบจริง 36 เดือน (พ.ค. 2023 – เม.ย. 2026) · ข้างหน้า = พยากรณ์จากข้อมูลถึง เม.ย. 2026 พร้อมช่วงความไม่แน่นอน 80%")
    cc1, cc2, cc3 = st.columns([1.2, 1.2, 1])
    with cc1:
        isos = sorted(D["stat"].iso2.unique(), key=cname); iso = st.selectbox("ประเทศ/ตลาด", isos, index=isos.index("US") if "US" in isos else 0, format_func=cname)
    with cc2:
        hss = sorted(D["stat"][D["stat"].iso2 == iso].hs4.unique()); hs = st.selectbox("กลุ่มสินค้า (HS4)", hss, index=hss.index("1604") if "1604" in hss else 0, format_func=hs_label)
    with cc3:
        h = st.radio("ระยะที่ใช้แสดงผลย้อนหลัง", [1, 3, 6], horizontal=True, format_func=lambda v: f"{v} เดือน")
    o1, o2 = st.columns(2); show_band = o1.checkbox("แสดงช่วง 80%", True); show_ses = o2.checkbox("แสดง baseline SES (เปรียบเทียบ)", False)

    g = group_of(iso); icon, msg = RELIABILITY[g]
    x = series_values(hs, iso); bt = D["bt"][(D["bt"].iso2 == iso) & (D["bt"].hs4 == hs) & (D["bt"].h == h)].sort_values("target_month"); fw = D["fwd"][(D["fwd"].iso2 == iso) & (D["fwd"].hs4 == hs)].sort_values("h")
    hist = D["hist"][(D["hist"].iso2 == iso) & (D["hist"].hs4 == hs)].sort_values("date").tail(54)
    st.markdown(f"**{icon} กลุ่มตลาด {GROUP_NAMES[g]}** — {msg}")
    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist.value_usd / 1e6, mode="lines", name="ค่าจริง", line=dict(color="#111827", width=2.2))
    if len(bt):
        if show_band:
            fig.add_scatter(x=pd.concat([bt.target_month, bt.target_month[::-1]]), y=pd.concat([bt.hi80, bt.lo80[::-1]]) / 1e6, fill="toself", fillcolor="rgba(37,99,235,.15)", line=dict(width=0), name=f"ช่วง 80% (ย้อนหลัง {h} เดือน)", hoverinfo="skip")
        fig.add_scatter(x=bt.target_month, y=bt.pred_lightgbm / 1e6, mode="lines", name=f"LightGBM (ย้อนหลัง, ล่วงหน้า {h} เดือน)", line=dict(color="#2563eb", width=1.8))
        if show_ses: fig.add_scatter(x=bt.target_month, y=bt.pred_ses / 1e6, mode="lines", name="SES (baseline)", line=dict(color="#0d9488", width=1.4, dash="dot"))
    fig.add_scatter(x=fw.target_month, y=fw.pred / 1e6, mode="markers", name="พยากรณ์ข้างหน้า", marker=dict(size=13, symbol="diamond", color="#dc2626"),
                    error_y=dict(type="data", symmetric=False, array=(fw.hi80 - fw.pred) / 1e6 if show_band else None, arrayminus=(fw.pred - fw.lo80) / 1e6 if show_band else None, color="#dc2626", thickness=1.6))
    fig.add_vline(x=LAST, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(height=470, template="plotly_white", yaxis_title="ล้านดอลลาร์สหรัฐ", title=f"{cname(iso)} · {hs_label(hs)}", legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fig, width="stretch")

    m1, m2, m3 = st.columns(3)
    if len(bt):
        wl = (bt.pred_lightgbm - bt.actual).abs().sum() / bt.actual.sum(); ws = (bt.pred_ses - bt.actual).abs().sum() / bt.actual.sum(); cov = ((bt.actual >= bt.lo80) & (bt.actual <= bt.hi80)).mean()
        m1.metric(f"WAPE ย้อนหลัง {h} เดือน (LightGBM)", f"{wl:.1%}", f"SES (เทียบ): {ws:.1%}", delta_color="off", delta_arrow="off"); m2.metric("ช่วง 80% ครอบคลุมค่าจริง", f"{cov:.0%}", "เป้า 80%", delta_color="off", delta_arrow="off")
        gl = D["bt"][(D["bt"].h == h)].assign(g=lambda d: d.iso2.map(group_of)); gg = gl[gl.g == g]; m3.metric(f"ผิดพลาดเฉลี่ยของ {'กลุ่ม ' + g}", f"{(gg.pred_lightgbm - gg.actual).abs().sum() / gg.actual.sum():.1%}", "ทั้งกลุ่ม (ถ่วงมูลค่า)", delta_color="off", delta_arrow="off")
    last12 = x[-12:].mean() / 1e6
    tb = fw.assign(เดือน=fw.target_month.dt.strftime("%b %Y"), ระยะ=fw.h.map(lambda v: f"{v} เดือนข้างหน้า"))
    tb = pd.DataFrame({"ระยะ": tb.ระยะ, "เดือนเป้าหมาย": tb.เดือน, "พยากรณ์ (ล้าน USD)": tb.pred / 1e6, "ช่วง 80% ล่าง": tb.lo80 / 1e6, "ช่วง 80% บน": tb.hi80 / 1e6, "SES (เทียบ)": tb.pred_ses / 1e6, "ค่าเฉลี่ย 12 เดือน": tb.pred_ma12 / 1e6, "เทียบค่าเฉลี่ย 12 เดือนล่าสุด": (tb.pred / 1e6 / last12 - 1) * 100})
    st.subheader("พยากรณ์ล่วงหน้า (ยังไม่รู้ผลจริง)")
    st.dataframe(tb, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.2f") for c in tb.columns[2:7]} | {"เทียบค่าเฉลี่ย 12 เดือนล่าสุด": st.column_config.NumberColumn(format="%+.0f%%")})
    if g in ("B", "C", "D", "นอกกลุ่ม"): st.warning("ตลาดกลุ่มนี้ผิดพลาดสูงในการทดสอบย้อนหลัง (42–70% ของมูลค่า) ช่วง 80% กว้างมากและแทบไม่ให้ข้อมูล ใช้เป็นภาพรวมเท่านั้น")
    if h == 6: st.info("ที่ระยะ 6 เดือน โมเดลไม่ได้ดีกว่า SES/ค่าเฉลี่ย 12 เดือนอย่างมีนัยสำคัญ (ดู 'วิธีอ่านและข้อจำกัด')")

# ================================================================================================ 4) Early Warning
elif PAGE.startswith("🚨"):
    st.title("🚨 Early Warning: ตลาดที่ยอดอาจตกใน 6 เดือนข้างหน้า")
    st.caption("ความน่าจะเป็นที่ *ยอดรวม 12 เดือน* ของตลาด (ประเทศ × กลุ่มสินค้า) จะตกเกิน 20% เทียบ 12 เดือนก่อนหน้า ภายใน 6 เดือน — เฉพาะตลาดที่ยอดฐานปีก่อน ≥ 1 ล้านดอลลาร์ · โมเดล LightGBM · ข้อมูลถึง เม.ย. 2026 (ผลจริงรู้ราว ต.ค. 2026)")
    ew = D["ew"].copy(); ew["กลุ่มตลาด"] = ew.iso2.map(group_of); ew["ประเทศ"] = ew.iso2.map(cname); ew["สินค้า"] = ew.hs4.map(hs_label)
    f1, f2, f3 = st.columns([1.5, 1.2, 1.1])
    hsf = f1.multiselect("กลุ่มสินค้า", sorted(ew.hs4.unique()), format_func=hs_label); grf = f2.multiselect("กลุ่มตลาด", ["A", "B", "C", "D", "นอกกลุ่ม"], format_func=lambda v: GROUP_NAMES[v]); thr = f3.slider("แสดงเฉพาะความเสี่ยง ≥", 0.0, 1.0, 0.5, 0.05)
    v = ew[(ew.risk >= thr)]
    if hsf: v = v[v.hs4.isin(hsf)]
    if grf: v = v[v.กลุ่มตลาด.isin(grf)]
    v = v.sort_values("risk", ascending=False).reset_index(drop=True)
    k1, k2, k3 = st.columns(3); k1.metric("ตลาดที่อยู่ในเกณฑ์ประเมิน", f"{len(ew)}"); k2.metric(f"ความเสี่ยง ≥ {thr:.0%}", f"{(ew.risk >= thr).sum()}", f"{(ew.risk >= thr).mean():.0%} ของตลาด", delta_color="off", delta_arrow="off"); k3.metric("ฐานยอดรวมของตลาดที่ถูกเตือน (USD/ปี)", f"{v.base.sum() / 1e6:,.0f} ล้าน")
    st.dataframe(pd.DataFrame({"อันดับ": v.index + 1, "ประเทศ": v.ประเทศ, "สินค้า": v.สินค้า, "กลุ่มตลาด": v.กลุ่มตลาด, "ความน่าจะเป็นที่ยอดตก > 20%": v.risk * 100, "YoY ปัจจุบัน": v.yoy_cur * 100, "YoY ที่คาด (run-rate)": v.proj3 * 100, "ยอดฐาน 12 เดือน (ล้าน USD)": v.base / 1e6}),
                 hide_index=True, width="stretch", height=420, column_config={"ความน่าจะเป็นที่ยอดตก > 20%": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100), "YoY ปัจจุบัน": st.column_config.NumberColumn(format="%+.0f%%"),
                                                                                     "YoY ที่คาด (run-rate)": st.column_config.NumberColumn(format="%+.0f%%"), "ยอดฐาน 12 เดือน (ล้าน USD)": st.column_config.NumberColumn(format="%.1f")})
    st.caption("YoY = การเปลี่ยนแปลงของยอดรวม 12 เดือนเทียบปีก่อน (ค่า -100% ถึง +500% ตัดที่ขอบ) · run-rate = ยอด 6 เดือนล่าสุดที่รู้แล้ว + 6 เดือนข้างหน้าด้วยค่าเฉลี่ย 3 เดือนล่าสุด")
    if len(v):
        st.subheader("ดูรายละเอียดตลาด")
        pick = st.selectbox("เลือกตลาด", range(min(len(v), 100)), format_func=lambda i: f"{i + 1}. {v.ประเทศ[i]} · {v.สินค้า[i]} ({v.risk[i]:.0%})"); r = v.iloc[pick]; x = series_values(r.hs4, r.iso2)
        h24 = D["hist"][(D["hist"].iso2 == r.iso2) & (D["hist"].hs4 == r.hs4)].sort_values("date").tail(30); m3 = x[-3:].mean() / 1e6
        fut = pd.date_range(LAST + pd.DateOffset(months=1), periods=6, freq="MS"); fig = go.Figure()
        fig.add_scatter(x=h24.date, y=h24.value_usd / 1e6, mode="lines+markers", name="ยอดจริงรายเดือน", line=dict(color="#111827"), marker=dict(size=5))
        fig.add_scatter(x=[LAST] + list(fut), y=[x[-1] / 1e6] + [m3] * 6, mode="lines", name="run-rate: ถ้าค่าเฉลี่ย 3 เดือนล่าสุดคงที่", line=dict(color="#dc2626", dash="dash"))
        fig.add_vline(x=LAST, line_dash="dash", line_color="#9ca3af"); fig.update_layout(height=380, template="plotly_white", yaxis_title="ล้านดอลลาร์/เดือน", title=f"{r.ประเทศ} · {r.สินค้า}", legend=dict(orientation="h", y=-0.2)); st.plotly_chart(fig, width="stretch")
        n1, n2, n3 = st.columns(3); n1.metric("ความน่าจะเป็นที่ยอดตก > 20%", f"{r.risk:.0%}"); n2.metric("YoY ปัจจุบัน", f"{r.yoy_cur:+.0%}"); n3.metric("YoY ที่คาด (run-rate)", f"{r.proj3:+.0%}")
        if r.yoy_cur > -0.2 and r.proj3 < -0.3: st.info("📌 ตลาดนี้ **ยังไม่ตกในยอดรวม 12 เดือน แต่ยอดล่าสุดอ่อนลงมากแล้ว** — ตรงกับกรณีที่ Early Warning ออกแบบมาเตือนก่อนตกจริง")
    with st.expander("วิธีอ่านผลนี้อย่างถูกต้อง"):
        st.markdown("""
- ในการทดสอบย้อนหลัง 31 เดือน ถ้าเตือนตลาดที่เสี่ยงสุด **10%** ต่อเดือน ตลาดที่เตือนตกจริง ~89% และเตือน 20% ได้ ~77% (อัตราตลาดตกทั่วไป ~22%)
- ความน่าจะเป็นค่อนข้างตรงกับความจริง (ประเมินสูงเล็กน้อย) แต่ **ไม่ใช่การรับประกัน** ใช้เป็นรายการ "ตรวจสอบเพิ่ม" ร่วมกับความรู้ของผู้เชี่ยวชาญ
- ส่วนใหญ่ของความแม่นมาจากการที่ครึ่งหนึ่งของหน้าต่าง 12 เดือนรู้ผลแล้ว โมเดล ML เพิ่มจากกฎ run-rate ราว 4–5%
- เกณฑ์ (ตก > 20%, 6 เดือน, ฐาน ≥ 1 ล้านดอลลาร์) ตั้งล่วงหน้า ยังไม่ได้กำหนดเกณฑ์ตัดสินใจตามต้นทุนจริง""")

# ================================================================================================ 5) สถานการณ์จำลอง
elif PAGE.startswith("🧪"):
    st.title("🧪 สถานการณ์จำลอง (What-if)")
    st.caption("ลองปรับยอดส่งออกของเดือนล่าสุด แล้วให้ *โมเดลจริง* คำนวณพยากรณ์และความเสี่ยงใหม่ทันที (ตัวแปรทั้งหมดถูกคำนวณใหม่จากประวัติที่แก้ไข)")
    sc1, sc2 = st.columns([1, 1.3])
    with sc1: iso, hs = series_picker("wi")
    with sc2:
        n_m = st.slider("ปรับยอดย้อนหลังกี่เดือนล่าสุด", 1, 6, 3); mult = st.slider("ตัวคูณยอด (100% = ไม่เปลี่ยน)", 10, 200, 70, 5, format="%d%%")
    x0 = series_values(hs, iso); x1 = x0.copy(); x1[-n_m:] = x1[-n_m:] * mult / 100
    st0 = static_of(hs, iso); ff0, p0 = forecast_series(x0, st0); ff1, p1 = forecast_series(x1, {**st0, "x_orig": x0})
    ew0, ew1 = core.ew_features(x0, ff0), core.ew_features(x1, ff1)
    in_pop = ew0["_base"] >= core.EW_MIN_BASE
    risk0, risk1 = (core.predict_ew(MODELS, META, [ew0])[0], core.predict_ew(MODELS, META, [ew1])[0]) if in_pop else (np.nan, np.nan)
    hist = D["hist"][(D["hist"].iso2 == iso) & (D["hist"].hs4 == hs)].sort_values("date").tail(30)
    tgt = {h: LAST + pd.DateOffset(months=h) for h in core.HORIZONS}
    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist.value_usd / 1e6, mode="lines", name="ประวัติจริง", line=dict(color="#111827", width=2))
    mod_idx = hist.date.iloc[-n_m:]; fig.add_scatter(x=mod_idx, y=x1[-n_m:] / 1e6, mode="lines+markers", name=f"สมมติ ({mult}% ของยอด {n_m} เดือนล่าสุด)", line=dict(color="#f97316", width=2.4))
    fig.add_scatter(x=[tgt[h] for h in core.HORIZONS], y=[p0.loc[h, "pred"] / 1e6 for h in core.HORIZONS], mode="markers", name="พยากรณ์ — ข้อมูลจริง", marker=dict(size=12, symbol="diamond", color="#2563eb"),
                    error_y=dict(type="data", symmetric=False, array=[(p0.loc[h, "hi80"] - p0.loc[h, "pred"]) / 1e6 for h in core.HORIZONS], arrayminus=[(p0.loc[h, "pred"] - p0.loc[h, "lo80"]) / 1e6 for h in core.HORIZONS], color="#93c5fd"))
    fig.add_scatter(x=[tgt[h] + pd.Timedelta(days=4) for h in core.HORIZONS], y=[p1.loc[h, "pred"] / 1e6 for h in core.HORIZONS], mode="markers", name="พยากรณ์ — ตามสถานการณ์สมมติ", marker=dict(size=12, symbol="diamond", color="#dc2626"),
                    error_y=dict(type="data", symmetric=False, array=[(p1.loc[h, "hi80"] - p1.loc[h, "pred"]) / 1e6 for h in core.HORIZONS], arrayminus=[(p1.loc[h, "pred"] - p1.loc[h, "lo80"]) / 1e6 for h in core.HORIZONS], color="#fca5a5"))
    fig.add_vline(x=LAST, line_dash="dash", line_color="#9ca3af"); fig.update_layout(height=460, template="plotly_white", yaxis_title="ล้านดอลลาร์/เดือน", title=f"{cname(iso)} · {hs_label(hs)}", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    tb = pd.DataFrame({"ระยะ": [f"{h} เดือน" for h in core.HORIZONS], "พยากรณ์จริง (ล้าน USD)": [p0.loc[h, "pred"] / 1e6 for h in core.HORIZONS], "พยากรณ์สมมติ (ล้าน USD)": [p1.loc[h, "pred"] / 1e6 for h in core.HORIZONS],
                       "เปลี่ยนแปลง": [(p1.loc[h, "pred"] / max(p0.loc[h, "pred"], 1) - 1) * 100 for h in core.HORIZONS], "ช่วง 80% สมมติ": [f"{p1.loc[h, 'lo80'] / 1e6:.2f} – {p1.loc[h, 'hi80'] / 1e6:.2f}" for h in core.HORIZONS]})
    st.dataframe(tb, hide_index=True, width="stretch", column_config={"พยากรณ์จริง (ล้าน USD)": st.column_config.NumberColumn(format="%.2f"), "พยากรณ์สมมติ (ล้าน USD)": st.column_config.NumberColumn(format="%.2f"), "เปลี่ยนแปลง": st.column_config.NumberColumn(format="%+.0f%%")})
    e1, e2, e3 = st.columns(3)
    if in_pop:
        e1.metric("ความเสี่ยงยอดตก > 20% — จริง", f"{risk0:.0%}"); e2.metric("ความเสี่ยง — ตามสมมติ", f"{risk1:.0%}", f"{(risk1 - risk0) * 100:+.0f} จุด", delta_color="inverse"); e3.metric("YoY ที่คาด (run-rate) — สมมติ", f"{ew1['proj3']:+.0%}", f"จริง {ew0['proj3']:+.0%}", delta_color="off", delta_arrow="off")
    else: e1.info("ตลาดนี้ยอดฐานปีก่อน < 1 ล้านดอลลาร์ ไม่อยู่ในเกณฑ์ประเมิน Early Warning")
    st.warning("⚠️ นี่คือการ *ทดลองว่าโมเดลตอบสนองต่อยอดล่าสุดอย่างไร* ไม่ใช่การพยากรณ์เหตุการณ์จริง (โมเดลเรียนรู้จากข้อมูลย้อนหลัง ไม่รู้สาเหตุเชิงเหตุผลของยอดที่ตก เช่น ภาษี หรือการห้ามนำเข้า)")

# ================================================================================================ 6) วิธีอ่านและข้อจำกัด
else:
    st.title("ℹ️ วิธีอ่านผลและข้อจำกัด")
    st.markdown(f"""
### แต่ละหน้าบอกอะไร
- **ภาพรวม** — ตัวเลขสำคัญและยอดส่งออกรวมย้อนหลัง/พยากรณ์
- **กลุ่มตลาด** — จัดประเทศเป็น 4 กลุ่มตามพฤติกรรม (K-Means) เพื่อบอกว่าตลาดไหนควรเชื่อถือตัวเลขแค่ไหน
- **พยากรณ์** — ค่าพยากรณ์ 1/3/6 เดือนข้างหน้ารายตลาด-สินค้า พร้อมช่วง 80% และผลทดสอบย้อนหลัง 36 เดือน
- **Early Warning** — ความน่าจะเป็นที่ยอดรวม 12 เดือนจะตกเกิน 20% ภายใน 6 เดือน
- **สถานการณ์จำลอง** — ปรับยอดล่าสุดแล้วให้โมเดลคำนวณใหม่

### ความน่าเชื่อถือ (ทดสอบย้อนหลัง พ.ค. 2023 – เม.ย. 2026 ไม่เคยเห็นตอนเทรน)
| ระดับ | ความผิดพลาด (WAPE) 1 / 3 / 6 เดือน |
|---|---|
| ยอดส่งออกรวมทั้งประเทศ | ~{res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 1 เดือน · ระดับยอดรวม', 'WAPE'):.1%} / {res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 3 เดือน · ระดับยอดรวม', 'WAPE'):.1%} / {res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 6 เดือน · ระดับยอดรวม', 'WAPE'):.1%} |
| ระดับประเทศ | ~{res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 1 เดือน · ระดับประเทศ', 'WAPE'):.0%} / {res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 3 เดือน · ระดับประเทศ', 'WAPE'):.0%} / {res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 6 เดือน · ระดับประเทศ', 'WAPE'):.0%} |
| ตลาด-สินค้าแต่ละตัว (ภาพรวม) | ~{res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 1 เดือน · ระดับตลาด-สินค้า', 'WAPE'):.0%} / {res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 3 เดือน · ระดับตลาด-สินค้า', 'WAPE'):.0%} / {res('Goal 1 พยากรณ์', 'LightGBM', 'ระยะ 6 เดือน · ระดับตลาด-สินค้า', 'WAPE'):.0%} |
| ตลาดกลุ่ม A (92% ของมูลค่า) | ~19% / 22% / 23% |
| ตลาดกลุ่ม B–D | **42–70%** — ไม่ควรใช้ตัดสินใจรายตัว |

- **ช่วง 80%** ที่ปรับด้วย conformal ครอบคลุมค่าจริง ~82–84% ในการทดสอบ (ตลาดเล็กกว้างราว 2 เท่าของค่ากลาง)
- **ที่ 6 เดือน** LightGBM ไม่ได้ดีกว่า SES หรือค่าเฉลี่ย 12 เดือนอย่างมีนัยสำคัญ (WAPE 0.256 เท่ากัน)
- **LightGBM ดีกว่าวิธีง่าย ๆ เพียงราว 5%** (ที่ 1–3 เดือน) ส่วนใหญ่ของความแม่นยำมาจากการเรียบข้อมูลที่ดี ไม่ใช่ข้อมูลภายนอก (ค่าเงิน/เศรษฐกิจ ไม่ช่วย)

### ข้อจำกัดที่ต้องรู้
1. ทดสอบหน้าต่างเดียว 36 เดือน ผลอาจเปลี่ยนเมื่อสภาพการค้าเปลี่ยน (ภาษี การห้ามนำเข้า โรคระบาด) ซึ่งโมเดล **ไม่มีข้อมูลเหตุการณ์**
2. ข้อมูลเป็นรายงานศุลกากรไทยใน UN Comtrade ระดับ HS 4 หลัก (10 กลุ่ม) หน่วย USD ไต้หวันถูกรวมใน "Other Asia, nes" จึงไม่มีในรายการ
3. ครอบคลุม 332 อนุกรมที่ค้าขายต่อเนื่อง (~97% ของมูลค่า) ตลาดใหม่หรือประปรายพยากรณ์ไม่ได้
4. Early Warning: หน้าต่างของป้ายซ้อนกับตัวแปร จึงมี baseline กฎ run-rate ที่แข็ง และ ML เพิ่มราว 4–5% เท่านั้น
5. baseline SES เพิ่มหลังเห็นผลทดสอบรอบแรก (เปิดเผยแล้วในรายงาน) และไม่ได้ปรับค่านัยสำคัญสำหรับการเปรียบเทียบหลายครั้ง
6. ค่าพยากรณ์ข้างหน้ายังไม่รู้ผลจริง ผลของ Early Warning ณ เม.ย. 2026 จะรู้ราว ต.ค. 2026

### การใช้งานจริง (ข้อเสนอแนะ)
เทรนโมเดลใหม่ทุก 6–12 เดือน · ปรับเทียบช่วง 80% ใหม่อย่างน้อยปีละครั้ง · ติดตามความผิดพลาดและ coverage รายเดือนเทียบ SES

### ที่มาและการทำซ้ำ
ผลทั้งหมดมาจาก `main.ipynb` (CRISP-DM) และแอปนี้ใช้โมเดลที่เทรนด้วยข้อมูลถึง {LAST:%b %Y} (สร้างด้วย `scripts/build_app_assets.py` และทดสอบว่าตรงกับ notebook ด้วย `scripts/test_app_features.py`) ข้อมูล: UN Comtrade (ข้อมูลเปิด) · ข้อมูลประกอบ: World Bank, ECB""")
    st.divider(); st.caption("พัฒนาเป็นส่วนหนึ่งของโครงงานวิชา Machine Learning — ข้อมูลเปิดทั้งหมด ผลลัพธ์เพื่อการศึกษา")
