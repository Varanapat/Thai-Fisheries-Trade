"""
ดึงข้อมูลส่งออกสินค้าประมงรายเดือน x ประเทศปลายทาง x HS code
จาก MOC Open Data API (กระทรวงพาณิชย์)
Docs: https://data.moc.go.th/OpenData/ExportHarmonizeCountries

การใช้งาน:
    pip install requests pandas
    python fetch_moc_fish_exports.py --check                  # ยิงทดสอบ 1 request ดูหน้าตาข้อมูล
    python fetch_moc_fish_exports.py --start 2021 --end 2025  # ดึงจริง
    python fetch_moc_fish_exports.py --flow import            # ฝั่งนำเข้า (ถ้า endpoint ใช้ได้)

Output: data/moc_<flow>_fish_<start>_<end>.csv (long format: 1 แถว = เดือน x ประเทศ x HS)
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://dataapi.moc.go.th"
ENDPOINTS = {
    "export": "export-harmonize-countries",
    "import": "import-harmonize-countries",  # docs อ้างถึงแต่ไม่ได้อธิบายละเอียด ต้องลองยิงดู
}

# HS code สินค้าประมงหลัก (4 หลัก) ปรับเพิ่ม/ลดได้
# ใช้ชุด "พิกัดสินค้าประมง" ของกรมประมงเช็ก/ขยายรายการได้
HS_CODES = {
    "0302": "ปลาสด/แช่เย็น",
    "0303": "ปลาแช่แข็ง",
    "0304": "เนื้อปลา/ฟิลเล",
    "0306": "สัตว์น้ำมีเปลือก (กุ้ง ปู)",
    "0307": "หอย/หมึก",
    "1604": "ปลาปรุงแต่ง/ทูน่ากระป๋อง",
    "1605": "กุ้ง/สัตว์น้ำมีเปลือกปรุงแต่ง",
}

LIMIT = 1000          # docs ไม่ระบุค่าสูงสุด ถ้าได้แถวเท่ากับ LIMIT พอดี แปลว่าอาจโดนตัด ให้เพิ่มค่า
SLEEP_SEC = 1.0       # เว้นระยะระหว่าง request กันโดน block
TIMEOUT = 90          # API ช้ามาก 30 วิไม่พอ
HEADERS = {"User-Agent": "Mozilla/5.0 (student research; CRISP-DM project)",
           "Accept": "application/json"}
CACHE_DIR = Path("data/cache")
OUT_DIR = Path("data")


def fetch_one(flow: str, year: int, month: int, hs_code: str, retries: int = 3) -> list[dict]:
    """ยิง 1 request (1 เดือน x 1 HS) พร้อม cache ลงไฟล์ ดึงซ้ำจะไม่ยิง API ใหม่"""
    cache_file = CACHE_DIR / flow / f"{hs_code}_{year}_{month:02d}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    url = f"{BASE_URL}/{ENDPOINTS[flow]}"
    params = {"year": year, "month": month, "hs_code": hs_code, "limit": LIMIT}

    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and "error" in data:
                # API ตอบ 200 แต่ body เป็น error -> ห้าม cache ไม่งั้นรันใหม่จะได้ [] ตลอด
                print(f"  ! {hs_code} {year}-{month:02d}: API error {data['error']}")
                return []
            if not isinstance(data, list):   # บางครั้ง API อาจห่อ data ไว้ใน key อื่น
                data = data.get("data", []) if isinstance(data, dict) else []
            if len(data) >= LIMIT:
                print(f"  ! {hs_code} {year}-{month:02d}: ได้ {len(data)} แถว = LIMIT อาจโดนตัด")
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except (requests.RequestException, ValueError) as e:
            print(f"  ! {hs_code} {year}-{month:02d} attempt {attempt}: {e}")
            time.sleep(2 * attempt)
    return []


def month_range(start_year: int, end_year: int):
    """ไล่เดือน ข้ามเดือนที่ยังไม่มีข้อมูล (API อัปเดตหลังวันที่ 20 ของเดือนถัดไป)"""
    today = date.today()
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            months_ago = (today.year - y) * 12 + (today.month - m)
            if months_ago < 1 or (months_ago == 1 and today.day <= 20):
                continue
            yield y, m


def build_dataset(flow: str, start_year: int, end_year: int) -> pd.DataFrame:
    rows = []
    jobs = [(y, m, hs) for y, m in month_range(start_year, end_year) for hs in HS_CODES]
    print(f"จะยิงทั้งหมด {len(jobs)} requests (ที่ cache แล้วจะข้าม)")

    for i, (y, m, hs) in enumerate(jobs, 1):
        cached = (CACHE_DIR / flow / f"{hs}_{y}_{m:02d}.json").exists()
        records = fetch_one(flow, y, m, hs)
        for rec in records:
            rec["hs_code"] = hs
            rec["hs_desc_th"] = HS_CODES[hs]
            rows.append(rec)
        if not cached:
            time.sleep(SLEEP_SEC)
        if i % 50 == 0:
            print(f"  {i}/{len(jobs)} เสร็จ, สะสม {len(rows):,} แถว")

    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    num_cols = ["quantity", "acc_quantity", "value_usd", "acc_value_usd",
                "value_baht", "acc_value_baht"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))

    # docs: quantity = 0 เมื่อ HS นั้นมีหลายหน่วยวัดปนกัน ให้เป็น NaN แทน จะได้ไม่ทำให้ค่าเพี้ยน
    if "quantity" in df.columns:
        df.loc[df["quantity"] == 0, "quantity"] = pd.NA
        # unit value (บาท/หน่วย มักเป็น kg) เป็น feature ที่ใช้บอกระดับราคาของตลาด
        df["unit_value_baht"] = df["value_baht"] / df["quantity"]

    df = df.drop_duplicates(subset=["date", "hs_code", "country_code"])
    return df.sort_values(["hs_code", "country_code", "date"]).reset_index(drop=True)


def summarize(df: pd.DataFrame):
    print("\n===== สรุป =====")
    print(f"จำนวนแถว: {len(df):,}")
    print(f"ช่วงเวลา: {df['date'].min():%Y-%m} ถึง {df['date'].max():%Y-%m}")
    print(f"จำนวนประเทศ: {df['country_code'].nunique()}")
    print("\nแถวต่อ HS code:")
    print(df.groupby("hs_code").size().to_string())

    # ดูว่า series (HS x ประเทศ) ไหนมีข้อมูลครบพอจะเอาไปทำ forecast
    n_months = df["date"].nunique()
    cover = df.groupby(["hs_code", "country_code"])["date"].nunique() / n_months
    print(f"\nseries ทั้งหมด (HS x ประเทศ): {len(cover)}")
    print(f"series ที่มีข้อมูล >= 80% ของเดือน: {(cover >= 0.8).sum()}")

    top = (df.groupby("country_name_en")["value_baht"].sum()
             .sort_values(ascending=False).head(10) / 1e6)
    print("\nTop 10 ประเทศ (มูลค่ารวม ล้านบาท):")
    print(top.round(1).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow", choices=ENDPOINTS, default="export")
    p.add_argument("--start", type=int, default=2021)
    p.add_argument("--end", type=int, default=date.today().year)
    p.add_argument("--check", action="store_true", help="ยิงทดสอบ 1 request แล้วจบ")
    args = p.parse_args()

    if args.check:
        url = f"{BASE_URL}/{ENDPOINTS[args.flow]}"
        r = requests.get(url, params={"year": 2024, "month": 6, "hs_code": "0306", "limit": 5},
                         timeout=30)
        print("status:", r.status_code)
        print(json.dumps(r.json(), ensure_ascii=False, indent=2)[:2000])
        return

    df = clean(build_dataset(args.flow, args.start, args.end))
    if df.empty:
        print("ไม่ได้ข้อมูลเลย ลองรัน --check ดูว่า API ตอบอะไร")
        return

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"moc_{args.flow}_fish_{args.start}_{args.end}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")  # utf-8-sig ให้ Excel อ่านไทยได้
    print(f"\nบันทึกแล้ว: {out}")
    summarize(df)


if __name__ == "__main__":
    main()