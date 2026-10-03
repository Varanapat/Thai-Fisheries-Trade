"""สร้างไฟล์ที่แอป Streamlit ต้องใช้ -> app/data/ และ app/models/

  1. เทรนโมเดลสุดท้าย (LightGBM point + quantile 10/90 ต่อระยะ 1/3/6 เดือน + Early Warning) ด้วยข้อมูลทั้งหมดที่รู้ผลแล้ว
  2. คำนวณพยากรณ์ล่วงหน้าจาก origin ล่าสุด (เม.ย. 2026) และคะแนน Early Warning
  3. ตรวจว่า pipeline นี้ให้ผลตรงกับที่ main.ipynb บันทึกไว้ (ทำนายย้อนหลังที่ origin ล่าสุดของช่วงทดสอบ แล้วเทียบ)

สูตรทุกอย่างเหมือน main.ipynb (ส่วน 3.3, 4.2–4.4) — รัน: python scripts/build_app_assets.py
"""
import json, shutil, sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, PREP, APP = ROOT / "DATASET", ROOT / "DATASET" / "prepared", ROOT / "app"
sys.path.insert(0, str(APP))
import core  # noqa: E402

rd = lambda p, **kw: pd.read_csv(p, keep_default_na=False, na_values=[""], encoding="utf-8-sig", **kw)
(APP / "data").mkdir(exist_ok=True); (APP / "models").mkdir(exist_ok=True)

# ---------- ข้อมูลดิบ -> เมทริกซ์ (เดือน × อนุกรม) ----------
cm = rd(DATA / "external/comtrade_thailand_seafood_monthly.csv", dtype={"hs4": str})
cm["date"] = pd.to_datetime(dict(year=cm.year, month=cm.month, day=1))
LAST = cm.date.max(); MONTHS = pd.date_range("2010-01-01", LAST, freq="MS")
part = cm[(~cm.is_world_total) & cm.partner_iso2.notna()]
V = part.pivot_table(index=["hs4", "partner_iso2"], columns="date", values="value_usd", aggfunc="sum").reindex(columns=MONTHS).fillna(0.0)
Wt = part.pivot_table(index=["hs4", "partner_iso2"], columns="date", values="net_weight_kg", aggfunc="sum").reindex(columns=MONTHS)
uni = rd(PREP / "series_universe.csv", dtype={"hs4": str})
universe = pd.MultiIndex.from_frame(uni)
Xs, Wd = V.loc[universe].T, Wt.loc[universe].T.fillna(0.0)
print("เดือนล่าสุด:", LAST.date(), "| อนุกรม:", Xs.shape[1])

# ---------- แผงตัวแปร (จาก notebook ส่วน 3) ----------
PN = rd(PREP / "forecast_panel.csv", dtype={"hs4": str}, parse_dates=["origin"])
PN["sin_m"], PN["cos_m"] = np.sin(2 * np.pi * PN.origin_month / 12), np.cos(2 * np.pi * PN.origin_month / 12)
NUMF = [c for c in PN.columns if c not in ("origin", "hs4", "iso2", "region", "income", "origin_month", "y1", "y3", "y6") and not c.startswith("base_")]
CATF = ["hs4", "region", "income"]
for c in CATF: PN[c] = PN[c].astype("category")
FC_COLS = NUMF + ["origin_month"] + CATF
BEST_P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
VARS = {"L1 · สัดส่วน · ถ่วง √ขนาด (ค่าจากระยะ 1)": dict(delta=True, weight="sqrt", obj="l1"), "L2 · สัดส่วน · ถ่วง √ขนาด": dict(delta=True, weight="sqrt", obj="regression"),
        "L1 · สัดส่วน · ไม่ถ่วง": dict(delta=True, weight=None, obj="l1")}
t36 = rd(PREP / "tuning_h3_h6.csv")
KW = {1: dict(delta=True, weight="sqrt", obj="l1")}
for h in (3, 6): KW[h] = VARS[t36[t36["h"] == h].sort_values("WAPE ช่วงตรวจสอบ").iloc[0]["รูปแบบ"]]
print("รูปแบบการเทรนต่อระยะ:", KW)

def fit_reg(tr, h, obj, weight="sqrt", alpha=None):
    yt = np.log1p(tr[f"y{h}"]) - tr.mean12
    w = np.sqrt(np.expm1(tr.mean12) + 1) if weight == "sqrt" else None
    kw = dict(objective=obj, verbose=-1, random_state=0, n_jobs=4, **BEST_P)
    if alpha is not None: kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw).fit(tr[FC_COLS], yt, sample_weight=w)

def labeled(h, T): return PN[(PN.origin + pd.DateOffset(months=h) <= T) & PN[f"y{h}"].notna()]

# ---------- ปรับช่วง 80% (CQR) ด้วยช่วงตรวจสอบ ----------
Q = rd(PREP / "quantiles.csv", dtype={"hs4": str}, parse_dates=["origin"]); Q["lo"], Q["hi"] = np.minimum(Q.lo, Q.hi), np.maximum(Q.lo, Q.hi)
def cqr_q(v, nom=0.8):
    s = np.maximum(np.log1p(v.lo) - np.log1p(v.y), np.log1p(v.y) - np.log1p(v.hi)); n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * nom) / n)))
QHAT = {h: cqr_q(Q[(Q.h == h) & (Q.split == "val")]) for h in core.HORIZONS}
print("CQR q-hat:", {h: round(v, 4) for h, v in QHAT.items()})

# ---------- 1) เทรนโมเดลสุดท้ายด้วยข้อมูลทั้งหมด + พยากรณ์ล่วงหน้า ----------
now_rows = PN[PN.origin == LAST].reset_index(drop=True)
fwd = []
for h in core.HORIZONS:
    tr = labeled(h, LAST)
    for kind, kw in (("point", dict(obj=KW[h]["obj"], weight=KW[h]["weight"])), ("q10", dict(obj="quantile", weight="sqrt", alpha=0.1)), ("q90", dict(obj="quantile", weight="sqrt", alpha=0.9))):
        m = fit_reg(tr, h, **kw); m.booster_.save_model(str(APP / "models" / f"lgbm_h{h}_{kind}.txt"))
        raw = np.expm1(m.predict(now_rows[FC_COLS]) + now_rows.mean12.to_numpy()); now_rows[f"{kind}_{h}"] = raw
    print(f"  h={h}: เทรนด้วย {len(tr):,} แถว (origin ถึง {(LAST - pd.DateOffset(months=h)).date()})")
    lo_r, hi_r = np.minimum(now_rows[f"q10_{h}"], now_rows[f"q90_{h}"]), np.maximum(now_rows[f"q10_{h}"], now_rows[f"q90_{h}"])
    pt = np.maximum(now_rows[f"point_{h}"], 0)
    lo = np.maximum(np.expm1(np.log1p(np.maximum(lo_r, 0)) - QHAT[h]), 0); hi = np.expm1(np.log1p(np.maximum(hi_r, 0)) + QHAT[h])
    ses_a = {1: 0.5, 3: 0.2, 6: 0.1}[h]                                                   # α ของ SES ที่เลือกจากช่วงตรวจสอบ (notebook ส่วน 4.3.2)
    ses = Xs.ewm(alpha=ses_a, adjust=False).mean().iloc[-1]
    fwd.append(pd.DataFrame({"hs4": now_rows.hs4.astype(str), "iso2": now_rows.iso2, "h": h, "origin": LAST, "target_month": LAST + pd.DateOffset(months=h),
                             "pred": pt, "lo80": lo, "hi80": hi, "pred_ses": ses.reindex(pd.MultiIndex.from_arrays([now_rows.hs4.astype(str), now_rows.iso2])).to_numpy(), "pred_ma12": now_rows.base_ma12}))
FWD = pd.concat(fwd, ignore_index=True); FWD.to_csv(APP / "data/forecast_latest.csv", index=False, encoding="utf-8-sig")
print("พยากรณ์ล่วงหน้า:", FWD.shape, "| ยอดรวมทั้งหมด (ล้านดอลลาร์) h=1/3/6:", {h: round(FWD[FWD.h == h].pred.sum() / 1e6, 1) for h in core.HORIZONS})

# ---------- 2) Early Warning ----------
EW_H, EW_THR = 6, 0.20
def ew_dataset():
    fw, bw, ftw = {}, None, None
    S12 = Xs.rolling(12).sum(); kn = Xs.rolling(12 - EW_H).sum(); m3, m12 = Xs.rolling(3).mean(), Xs.rolling(12).mean()
    base = S12.shift(12 - EW_H); nb = base.replace(0, np.nan)
    f = {"yoy_cur": S12 / S12.shift(12).replace(0, np.nan) - 1, "base_log": np.log1p(base), "known_ratio": kn / nb, "proj3": (kn + EW_H * m3) / nb - 1, "proj12": (kn + EW_H * m12) / nb - 1,
         "known_yoy": kn / kn.shift(12).replace(0, np.nan) - 1, "base_trend": base / S12.shift(12 - EW_H + 12).replace(0, np.nan) - 1, "S12_log": np.log1p(S12)}
    f = {k: v.replace([np.inf, -np.inf], np.nan).clip(-1, 5) if k not in ("base_log", "S12_log") else v for k, v in f.items()}
    fut = S12.shift(-EW_H)
    sel = Xs.index >= pd.Timestamp("2015-01-01"); n_ = int(sel.sum()); S_ = Xs.shape[1]
    D = pd.DataFrame({"origin": np.repeat(Xs.index[sel], S_), "hs4": np.tile(Xs.columns.get_level_values(0).astype(str), n_), "iso2": np.tile(Xs.columns.get_level_values(1), n_),
                      "base": base.loc[sel].to_numpy().ravel(), "fut": fut.loc[sel].to_numpy().ravel(), **{k: v.loc[sel].to_numpy().ravel() for k, v in f.items()}})
    D = D.merge(PN[["origin", "hs4", "iso2", "mom3_12", "std6", "std12", "active12", "since_last", "uv12", "share_ctry12", "share_hs12", "lag0", "mean3", "mean12"]].assign(hs4=lambda d: d.hs4.astype(str)), on=["origin", "hs4", "iso2"], how="left")
    D["yoy_fut"] = D.fut / D.base.where(D.base > 0) - 1; D["inpop"] = D.base >= core.EW_MIN_BASE
    D["y"] = np.where(D.yoy_fut.notna(), (D.yoy_fut < -EW_THR).astype(float), np.nan)
    D["hs4"] = pd.Categorical(D.hs4, categories=list(core.HS4_NAME))
    return D
E = ew_dataset()
EWP = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=4)
EW_COLS = core.EW_FEATS + ["hs4"]
def ew_train(T):
    tr = E[(E.origin + pd.DateOffset(months=EW_H) <= T) & E.y.notna() & E.inpop]
    return lgb.LGBMClassifier(**EWP).fit(tr[EW_COLS], tr.y), len(tr)
ewm, n_ew = ew_train(LAST); ewm.booster_.save_model(str(APP / "models/ew_lgbm.txt"))
en = E[(E.origin == LAST) & E.inpop].copy(); en["risk"] = ewm.predict_proba(en[EW_COLS])[:, 1]
en[["hs4", "iso2", "risk", "yoy_cur", "proj3", "proj12", "base", "known_ratio"]].assign(hs4=lambda d: d.hs4.astype(str)).to_csv(APP / "data/ew_latest.csv", index=False, encoding="utf-8-sig")
print(f"Early Warning: เทรนด้วย {n_ew:,} แถว | ตลาดในประชากร {len(en)} | ความน่าจะเป็น > 50%: {(en.risk > 0.5).sum()}")

# ---------- ข้อมูลประกอบแอป ----------
hist = Xs.copy(); hist.columns = [f"{a}|{b}" for a, b in hist.columns]
(hist.reset_index(names="date").melt(id_vars="date", var_name="k", value_name="value_usd").assign(hs4=lambda d: d.k.str.split("|").str[0], iso2=lambda d: d.k.str.split("|").str[1])
     [["hs4", "iso2", "date", "value_usd"]].to_csv(APP / "data/history.csv", index=False, encoding="utf-8-sig"))
S12 = Xs.rolling(12).sum().iloc[-1]; M3 = Xs.rolling(3).mean().iloc[-1]; W12 = Wd.rolling(12).sum().iloc[-1]
st = pd.DataFrame({"s12": S12, "m3": M3, "w12": W12}); st.index.names = ["hs4", "iso2"]; st = st.reset_index()
st["hs_s12_tot"] = st.groupby("hs4").s12.transform("sum"); st["ctry_s12_tot"] = st.groupby("iso2").s12.transform("sum")
st["hs_mean3_raw"] = st.groupby("hs4").m3.transform("sum"); st["ctry_mean3_raw"] = st.groupby("iso2").m3.transform("sum")
stat = st.merge(now_rows.assign(hs4=lambda d: d.hs4.astype(str))[["hs4", "iso2", "thb_chg3", "ccy_chg3", "gdppc_log", "pop_log", "gdp_growth", "inflation", "origin_month", "region", "income"]], on=["hs4", "iso2"])
stat.drop(columns=["s12", "m3"]).to_csv(APP / "data/series_static.csv", index=False, encoding="utf-8-sig")
dim = rd(DATA / "clean/trade/dim_country.csv")
dim.rename(columns={"country_iso2": "iso2", "country_th": "name_th"})[["iso2", "name_th", "name_en"]].to_csv(APP / "data/countries.csv", index=False, encoding="utf-8-sig")
for f in ("forecast_test_predictions.csv", "clusters_w2.csv", "cluster_features_w2.csv", "results_summary.csv"): shutil.copy(PREP / f, APP / "data" / f)

meta = {"last_month": str(LAST.date()), "forecast_cols": FC_COLS, "ew_cols": EW_COLS, "qhat": {str(h): v for h, v in QHAT.items()}, "ses_alpha": {"1": 0.5, "3": 0.2, "6": 0.1},
        "categories_fc": {c: [str(v) for v in PN[c].cat.categories] for c in CATF}, "categories_ew": {"hs4": list(core.HS4_NAME)},
        "train_kw": {str(h): KW[h] for h in KW}, "params": BEST_P, "ew_params": {k: v for k, v in EWP.items() if k not in ("verbose", "n_jobs")}, "ew_horizon": EW_H, "ew_threshold": EW_THR,
        "n_series": int(Xs.shape[1])}
(APP / "models/model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

# ---------- 3) ตรวจว่า pipeline นี้ตรงกับ notebook ----------
print("\\n== ตรวจเทียบกับผลใน notebook ==")
ft = rd(PREP / "forecast_test_predictions.csv", dtype={"hs4": str}, parse_dates=["origin", "target_month"])
T = pd.Timestamp(ft[ft.h == 1].origin.max())
m = fit_reg(labeled(1, T), 1, obj=KW[1]["obj"], weight=KW[1]["weight"]); te = PN[(PN.origin == T) & PN["y1"].notna()]
mine = pd.DataFrame({"hs4": te.hs4.astype(str).to_numpy(), "iso2": te.iso2.to_numpy(), "mine": np.maximum(np.expm1(m.predict(te[FC_COLS]) + te.mean12.to_numpy()), 0)})
nbk = ft[(ft.h == 1) & (ft.origin == T)][["hs4", "iso2", "pred_lightgbm"]].merge(mine, on=["hs4", "iso2"])
rel = (nbk.mine - nbk.pred_lightgbm).abs() / nbk.pred_lightgbm.clip(lower=1)
print(f"  พยากรณ์ระยะ 1 ที่ origin {T.date()}: เทียบ {len(nbk)} อนุกรม | ความต่างสัมพัทธ์สูงสุด {rel.max():.2e} | มัธยฐาน {rel.median():.2e}")
ewp = rd(PREP / "ew_predictions.csv", dtype={"hs4": str}, parse_dates=["origin"]); ewp = ewp[(ewp.model == "LightGBM") & (ewp.split == "test")]
T2 = ewp.origin.max(); m2, _ = ew_train(T2); te2 = E[(E.origin == T2) & E.y.notna() & E.inpop]
mine2 = pd.DataFrame({"hs4": te2.hs4.astype(str).to_numpy(), "iso2": te2.iso2.to_numpy(), "mine": m2.predict_proba(te2[EW_COLS])[:, 1]})
nbk2 = ewp[ewp.origin == T2][["hs4", "iso2", "p"]].merge(mine2, on=["hs4", "iso2"])
print(f"  Early Warning ที่ origin {T2.date()}: เทียบ {len(nbk2)} อนุกรม | ความต่างสูงสุด {(nbk2.mine - nbk2.p).abs().max():.2e}")
assert rel.max() < 1e-3 and (nbk2.mine - nbk2.p).abs().max() < 1e-3, "pipeline ไม่ตรงกับ notebook!"
print("✅ ตรงกับ notebook")
