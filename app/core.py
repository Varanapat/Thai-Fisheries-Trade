"""แกนกลางของแอป: สร้างตัวแปรจากประวัติรายเดือนของ 1 อนุกรม แล้วทำนายด้วยโมเดลที่เทรนไว้

สูตรทุกตัวต้อง *เหมือนกับ main.ipynb* (ส่วน 3.3 make_features และส่วน 4.4 ew_build) — มีสคริปต์ทดสอบ
scripts/test_app_features.py ยืนยันว่าผลตรงกับตารางที่ notebook สร้างไว้ทุกอนุกรม
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR = APP_DIR / "data", APP_DIR / "models"
HORIZONS = (1, 3, 6)
EW_FEATS = ["yoy_cur", "base_log", "known_ratio", "proj3", "proj12", "known_yoy", "base_trend", "S12_log", "mom3_12", "std6", "std12",
            "active12", "since_last", "uv12", "share_ctry12", "share_hs12", "lag0", "mean3", "mean12"]
EW_MIN_BASE = 1_000_000

HS4_NAME = {
    "0301": "ปลามีชีวิต", "0302": "ปลาสด/แช่เย็น", "0303": "ปลาแช่แข็ง", "0304": "เนื้อปลา/ฟิลเล", "0305": "ปลาแห้ง/รมควัน",
    "0306": "กุ้ง ปู (มีเปลือก)", "0307": "หอย หมึก", "0308": "สัตว์น้ำไม่มีกระดูกสันหลังอื่น ๆ", "1604": "ปลาปรุงแต่ง/กระป๋อง", "1605": "กุ้ง ปู หอย ปรุงแต่ง",
}


def load_meta() -> dict:
    return json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))


def load_models() -> dict:
    """โหลดโมเดล LightGBM ทั้งหมด (point/q10/q90 ต่อระยะ + Early Warning)"""
    m = {}
    for h in HORIZONS:
        for kind in ("point", "q10", "q90"):
            m[(h, kind)] = lgb.Booster(model_file=str(MODEL_DIR / f"lgbm_h{h}_{kind}.txt"))
    m["ew"] = lgb.Booster(model_file=str(MODEL_DIR / "ew_lgbm.txt"))
    return m


# ---------------------------------------------------------------------------------------------
def hist_features(x: np.ndarray, static: dict) -> dict:
    """ตัวแปรของ 'เดือนสุดท้าย' ของอนุกรม x (ค่าดิบ USD เรียงตามเดือนตั้งแต่ ม.ค. 2010)

    static ต้องมี: w12, hs_mean3_raw, ctry_mean3_raw, hs_s12_tot, ctry_s12_tot (ยอดรวมของกลุ่ม/ประเทศที่ *รวมอนุกรมนี้อยู่แล้ว*)
    และตัวแปรที่ไม่ขึ้นกับยอดของอนุกรม: thb_chg3, ccy_chg3, gdppc_log, pop_log, gdp_growth, inflation, origin_month, hs4, region, income
    ถ้า x ถูกแก้ (สถานการณ์จำลอง) ยอดรวมของกลุ่ม/ประเทศจะถูกปรับตามส่วนต่างของอนุกรมนี้ผ่าน static['x_orig']"""
    x = np.asarray(x, dtype=float)
    L = np.log1p(x)
    f = {f"lag{k}": float(L[-1 - k]) for k in (0, 1, 2, 5, 11)}
    for n in (3, 6, 12): f[f"mean{n}"] = float(np.log1p(x[-n:].mean()))
    f["std6"], f["std12"] = float(np.std(L[-6:], ddof=1)), float(np.std(L[-12:], ddof=1))
    f["mom3_12"] = f["mean3"] - f["mean12"]
    s12, s12_prev = float(x[-12:].sum()), float(x[-24:-12].sum())
    f["yoy12"] = float(np.log1p(s12) - np.log1p(s12_prev))
    f["active12"] = float((x[-12:] > 0).sum())
    pos = np.nonzero(x > 0)[0]
    f["since_last"] = float(len(x) - 1 - pos[-1]) if len(pos) else float(len(x))
    w12 = static["w12"]
    f["uv12"] = float(np.log(s12 / w12)) if s12 > 0 and w12 > 0 else np.nan
    # ยอดรวมกลุ่มสินค้า/ประเทศ: ปรับตามส่วนต่างของอนุกรมนี้ (ไม่มีส่วนต่าง = ค่าเดิม)
    xo = np.asarray(static.get("x_orig", x), dtype=float)
    d_mean3, d_s12 = float(x[-3:].mean() - xo[-3:].mean()), s12 - float(xo[-12:].sum())
    f["hs_mean3"] = float(np.log1p(static["hs_mean3_raw"] + d_mean3)); f["ctry_mean3"] = float(np.log1p(static["ctry_mean3_raw"] + d_mean3))
    hs_s12, ct_s12 = static["hs_s12_tot"] + d_s12, static["ctry_s12_tot"] + d_s12
    f["share_hs12"] = s12 / hs_s12 if hs_s12 > 0 else np.nan
    f["share_ctry12"] = s12 / ct_s12 if ct_s12 > 0 else np.nan
    for k in ("thb_chg3", "ccy_chg3", "gdppc_log", "pop_log", "gdp_growth", "inflation", "origin_month"):
        f[k] = static[k]
    f["sin_m"], f["cos_m"] = float(np.sin(2 * np.pi * static["origin_month"] / 12)), float(np.cos(2 * np.pi * static["origin_month"] / 12))
    f["hs4"], f["region"], f["income"] = static["hs4"], static["region"], static["income"]
    f["_s12"] = s12
    return f


def ew_features(x: np.ndarray, ff: dict, H: int = 6) -> dict:
    """ตัวแปร Early Warning (นิยามเดียวกับ ew_build ใน notebook): ป้าย = S12(t+H) ÷ S12(t+H-12) - 1"""
    x = np.asarray(x, dtype=float); n = len(x) - 1
    S12 = lambda end: float(x[end - 11:end + 1].sum())
    s12, s12_prev, base, base_prev = S12(n), S12(n - 12), S12(n - (12 - H)), S12(n - (12 - H) - 12)
    kn, kn_prev = float(x[-(12 - H):].sum()), float(x[n - 12 - (12 - H) + 1:n - 12 + 1].sum())
    m3, m12 = float(x[-3:].mean()), float(x[-12:].mean())
    nb = base if base > 0 else np.nan
    clip = lambda v: float(np.clip(v, -1, 5)) if np.isfinite(v) else np.nan
    e = {"yoy_cur": clip(s12 / s12_prev - 1) if s12_prev > 0 else np.nan, "base_log": float(np.log1p(base)),
         "known_ratio": clip(kn / nb), "proj3": clip((kn + H * m3) / nb - 1), "proj12": clip((kn + H * m12) / nb - 1),
         "known_yoy": clip(kn / kn_prev - 1) if kn_prev > 0 else np.nan, "base_trend": clip(base / base_prev - 1) if base_prev > 0 else np.nan,
         "S12_log": float(np.log1p(s12))}
    for k in ("mom3_12", "std6", "std12", "active12", "since_last", "uv12", "share_ctry12", "share_hs12", "lag0", "mean3", "mean12"): e[k] = ff[k]
    e["hs4"], e["_base"] = ff["hs4"], base
    return e


# ---------------------------------------------------------------------------------------------
def _frame(rows: list[dict], cols: list[str], categories: dict) -> pd.DataFrame:
    d = pd.DataFrame(rows)[cols]
    for c, cats in categories.items():
        if c in d: d[c] = pd.Categorical(d[c], categories=cats)
    return d


def predict_forecast(models: dict, meta: dict, feats: list[dict]) -> pd.DataFrame:
    """ทำนายทุกระยะ: ค่ากลาง + ช่วง 80% (quantile ปรับด้วย conformal)  คืน DataFrame แถวละ (อนุกรม, ระยะ)"""
    cols = meta["forecast_cols"]; X = _frame(feats, cols, meta["categories_fc"]); mean12 = np.array([f["mean12"] for f in feats])
    out = []
    for h in HORIZONS:
        pt = np.maximum(np.expm1(models[(h, "point")].predict(X) + mean12), 0)
        lo_r, hi_r = np.expm1(models[(h, "q10")].predict(X) + mean12), np.expm1(models[(h, "q90")].predict(X) + mean12)
        lo_r, hi_r = np.minimum(lo_r, hi_r), np.maximum(lo_r, hi_r); q = meta["qhat"][str(h)]
        lo = np.maximum(np.expm1(np.log1p(np.maximum(lo_r, 0)) - q), 0); hi = np.expm1(np.log1p(np.maximum(hi_r, 0)) + q)
        out.append(pd.DataFrame({"h": h, "pred": pt, "lo80": lo, "hi80": hi}))
    return pd.concat(out, ignore_index=True)


def predict_ew(models: dict, meta: dict, ews: list[dict]) -> np.ndarray:
    X = _frame(ews, meta["ew_cols"], meta["categories_ew"])
    return models["ew"].predict(X)
