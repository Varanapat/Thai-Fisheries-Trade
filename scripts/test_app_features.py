"""ทดสอบแอป: (1) โค้ดสร้างตัวแปรใน app/core.py ตรงกับตาราง notebook ทุกอนุกรม (2) การทำนายผ่าน core ตรงกับไฟล์ที่ build_app_assets สร้างไว้
(3) พยากรณ์ระยะ 3/6 เดือน (รูปแบบเทรนต่างจากระยะ 1) ตรงกับ forecast_test_predictions.csv — รัน: python scripts/test_app_features.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP, PREP = ROOT / "app", ROOT / "DATASET" / "prepared"
sys.path.insert(0, str(APP))
import core
rd = lambda p, **kw: pd.read_csv(p, keep_default_na=False, na_values=[""], encoding="utf-8-sig", **kw)

meta = core.load_meta(); models = core.load_models()
hist = rd(APP / "data/history.csv", dtype={"hs4": str}, parse_dates=["date"]); stat = rd(APP / "data/series_static.csv", dtype={"hs4": str})
PN = rd(PREP / "forecast_panel.csv", dtype={"hs4": str}, parse_dates=["origin"]); last = pd.Timestamp(meta["last_month"]); PN = PN[PN.origin == last].set_index(["hs4", "iso2"])
PN["sin_m"], PN["cos_m"] = np.sin(2 * np.pi * PN.origin_month / 12), np.cos(2 * np.pi * PN.origin_month / 12)
H = {k: g.sort_values("date").value_usd.to_numpy() for k, g in hist.groupby(["hs4", "iso2"])}
fwd = rd(APP / "data/forecast_latest.csv", dtype={"hs4": str}); ew = rd(APP / "data/ew_latest.csv", dtype={"hs4": str})

rows, ffs, ews, keys = [], [], [], []
for _, r in stat.iterrows():
    k = (r.hs4, r.iso2); x = H[k]; assert len(x) == 196
    ff = core.hist_features(x, r.to_dict()); ffs.append(ff); keys.append(k); ews.append(core.ew_features(x, ff))
num = [c for c in meta["forecast_cols"] if c not in ("hs4", "region", "income")]
diffs = {c: max(abs(f[c] - PN.loc[k, c]) if np.isfinite(f[c]) and np.isfinite(PN.loc[k, c]) else (0 if (np.isnan(f[c]) and pd.isna(PN.loc[k, c])) else 1e9) for f, k in zip(ffs, keys)) for c in num}
bad = {c: v for c, v in diffs.items() if v > 1e-6}
print(f"ตัวแปรพยากรณ์: เทียบ {len(ffs)} อนุกรม × {len(num)} ตัวแปร | ความต่างสูงสุด {max(diffs.values()):.2e} | ตัวที่เกิน 1e-6: {bad or 'ไม่มี'}")
assert not bad

fp = core.predict_forecast(models, meta, ffs); n = len(ffs)
fp["hs4"], fp["iso2"] = [k[0] for k in keys] * 3, [k[1] for k in keys] * 3
m = fp.merge(fwd, on=["hs4", "iso2", "h"], suffixes=("", "_f"))
for c in ("pred", "lo80", "hi80"): print(f"  ทำนายผ่าน core vs ไฟล์ ({c}): ความต่างสัมพัทธ์สูงสุด {((m[c] - m[c + '_f']).abs() / m[c + '_f'].clip(lower=1)).max():.2e}"); assert ((m[c] - m[c + '_f']).abs() / m[c + '_f'].clip(lower=1)).max() < 1e-6

# Early Warning: ตัวแปร + ความน่าจะเป็น
e = pd.DataFrame(ews); e["iso2"] = [k[1] for k in keys]; e["hs4"] = [k[0] for k in keys]
e = e[e._base >= core.EW_MIN_BASE].merge(ew, on=["hs4", "iso2"], suffixes=("", "_f"))
for c in ("yoy_cur", "proj3", "proj12", "known_ratio"): d = (e[c] - e[c + "_f"]).abs().max(); print(f"  EW ตัวแปร {c}: ต่างสูงสุด {d:.2e}"); assert d < 1e-6
e_rows = [dict(r) for _, r in e.iterrows()]
pe = core.predict_ew(models, meta, e_rows); d = np.abs(pe - e.risk.to_numpy()).max(); print(f"  EW ความน่าจะเป็น ผ่าน core vs ไฟล์: ต่างสูงสุด {d:.2e}"); assert d < 1e-6

# พยากรณ์ย้อนหลังระยะ 3/6: เทรนเหมือน build_app_assets ที่ origin ล่าสุดของช่วงทดสอบ แล้วเทียบ notebook
import lightgbm as lgb
PNf = rd(PREP / "forecast_panel.csv", dtype={"hs4": str}, parse_dates=["origin"]); PNf["sin_m"], PNf["cos_m"] = np.sin(2 * np.pi * PNf.origin_month / 12), np.cos(2 * np.pi * PNf.origin_month / 12)
for c in ("hs4", "region", "income"): PNf[c] = PNf[c].astype("category")
ft = rd(PREP / "forecast_test_predictions.csv", dtype={"hs4": str}, parse_dates=["origin", "target_month"])
for h in (3, 6):
    kw = meta["train_kw"][str(h)]; T = pd.Timestamp(ft[ft.h == h].origin.max()); tr = PNf[(PNf.origin + pd.DateOffset(months=h) <= T) & PNf[f"y{h}"].notna()]; te = PNf[(PNf.origin == T) & PNf[f"y{h}"].notna()]
    w = np.sqrt(np.expm1(tr.mean12) + 1) if kw["weight"] == "sqrt" else None
    mdl = lgb.LGBMRegressor(objective=kw["obj"], verbose=-1, random_state=0, n_jobs=4, **meta["params"]).fit(tr[meta["forecast_cols"]], np.log1p(tr[f"y{h}"]) - tr.mean12, sample_weight=w)
    mine = pd.DataFrame({"hs4": te.hs4.astype(str).to_numpy(), "iso2": te.iso2.to_numpy(), "mine": np.maximum(np.expm1(mdl.predict(te[meta["forecast_cols"]]) + te.mean12.to_numpy()), 0)})
    nb = ft[(ft.h == h) & (ft.origin == T)][["hs4", "iso2", "pred_lightgbm"]].merge(mine, on=["hs4", "iso2"]); rel = ((nb.mine - nb.pred_lightgbm).abs() / nb.pred_lightgbm.clip(lower=1)).max()
    print(f"  ระยะ {h} เดือน (origin {T.date()}, รูปแบบเทรน {kw}): ต่างสัมพัทธ์สูงสุด {rel:.2e}"); assert rel < 1e-3
print("✅ ผ่านทุกการทดสอบ")
