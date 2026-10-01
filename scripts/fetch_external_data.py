"""ดึงข้อมูลภายนอก (open data, ไม่ต้องใช้ API key) ลง DATASET/external/

  1. อัตราแลกเปลี่ยนรายวัน  : Frankfurter API (อัตราอ้างอิง ECB)
  2. ตัวชี้วัดเศรษฐกิจรายปี : World Bank Open Data API (WDI)
  3. สภาพอากาศรายวัน       : Open-Meteo Historical Weather API (ERA5)
  4. (fxlong, wblong) ชุดยาวสำหรับ Data Preparation: อัตราแลกเปลี่ยนรายเดือนตั้งแต่ 2010 และ World Bank ตั้งแต่ 2008
     (ชุดข้อมูลของ Comtrade เริ่มปี 2010 และ feature ใช้ค่าย้อนหลัง จึงต้องการประวัติยาวกว่าชุดข้อมูลแรก)

ใช้: python scripts/fetch_external_data.py            (ดึงทั้งหมด)
     python scripts/fetch_external_data.py fx wb wx   (เลือกบางชุด)
"""
import json, sys, time, urllib.request, urllib.parse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "DATASET" / "external"
OUT.mkdir(parents=True, exist_ok=True)
START = "2021-01-01"  # ครอบคลุมตั้งแต่ข้อมูลการค้าปี 2564 เป็นต้นไป


def get_json(url, params=None, retries=4):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "thai-fisheries-ml-project/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"  retry {i + 1}: {e}")
            time.sleep(5 * (i + 1))


def fetch_fx():
    rows = []
    today = date.today()
    for y in range(int(START[:4]), today.year + 1):
        end = min(date(y, 12, 31), today)
        js = get_json(f"https://api.frankfurter.dev/v1/{y}-01-01..{end}", {"base": "USD"})
        for d, rates in js["rates"].items():
            rows += [(d, cur, v) for cur, v in rates.items()]
        print("fx", y, len(js["rates"]), "days")
    df = pd.DataFrame(rows, columns=["date", "currency", "per_usd"])
    # ให้มีคู่ USD→THB ชัด ๆ และคำนวณ THB ต่อ 1 หน่วยสกุลนั้น
    thb = df[df.currency == "THB"].set_index("date").per_usd.rename("thb_per_usd")
    df = df.join(thb, on="date")
    df["thb_per_unit"] = df.thb_per_usd / df.per_usd
    df = df.sort_values(["currency", "date"])
    df.to_csv(OUT / "fx_daily_usd_base_ecb.csv", index=False, encoding="utf-8-sig")
    print("saved fx", df.shape)


WB_INDICATORS = {
    "NY.GDP.MKTP.CD": "gdp_usd",
    "NY.GDP.PCAP.CD": "gdp_per_capita_usd",
    "SP.POP.TOTL": "population",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_pct",
}


def fetch_wb():
    meta = get_json("https://api.worldbank.org/v2/country", {"format": "json", "per_page": 400})[1]
    m = pd.DataFrame([{
        "iso3": c["id"], "iso2": c["iso2Code"], "name_en": c["name"],
        "region": c["region"]["value"], "income_level": c["incomeLevel"]["value"],
        "capital": c["capitalCity"], "longitude": c["longitude"], "latitude": c["latitude"],
    } for c in meta])
    m["is_aggregate"] = m.region == "Aggregates"
    m.to_csv(OUT / "worldbank_country_meta.csv", index=False, encoding="utf-8-sig")
    print("saved country meta", m.shape)

    frames = []
    for code, name in WB_INDICATORS.items():
        js = get_json(f"https://api.worldbank.org/v2/country/all/indicator/{code}",
                      {"format": "json", "date": "2015:2025", "per_page": 20000})
        df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), name: r["value"]}
                           for r in js[1]])
        frames.append(df.set_index(["iso3", "year"]))
        print("wb", code, len(df), "rows; last updated", js[0]["lastupdated"])
    wide = pd.concat(frames, axis=1).reset_index()
    wide = wide.merge(m[["iso3", "iso2", "name_en", "is_aggregate"]], on="iso3", how="left")
    wide = wide[~wide.is_aggregate.fillna(True).astype(bool)].drop(columns="is_aggregate")
    wide = wide[["iso2", "iso3", "name_en", "year", *WB_INDICATORS.values()]].sort_values(["iso3", "year"])
    wide.to_csv(OUT / "worldbank_indicators_yearly.csv", index=False, encoding="utf-8-sig")
    print("saved wb", wide.shape)


# จังหวัดหลักที่เลี้ยงกุ้ง/มีท่าเรือประมง (พิกัดโดยประมาณของตัวจังหวัด)
LOCATIONS = {
    "สมุทรสาคร": (13.55, 100.27), "จันทบุรี": (12.61, 102.10), "ฉะเชิงเทรา": (13.69, 101.07),
    "นครศรีธรรมราช": (8.43, 99.96), "สุราษฎร์ธานี": (9.14, 99.33), "สงขลา": (7.19, 100.59),
}


def fetch_wx():
    end = (date.today() - timedelta(days=7)).isoformat()  # ERA5 ล่าช้าประมาณ 5 วัน
    frames = []
    for prov, (lat, lon) in LOCATIONS.items():
        js = get_json("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": lat, "longitude": lon, "start_date": START, "end_date": end,
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": "Asia/Bangkok"})
        df = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
        df.insert(1, "province", prov)
        df.insert(2, "latitude", lat)
        df.insert(3, "longitude", lon)
        frames.append(df)
        print("wx", prov, len(df))
        time.sleep(2)
    out = pd.concat(frames)
    out.to_csv(OUT / "weather_daily_shrimp_provinces.csv", index=False, encoding="utf-8-sig")
    print("saved wx", out.shape)


def fetch_fx_long():
    """ค่าเฉลี่ยรายเดือนของอัตราแลกเปลี่ยน (หน่วยสกุลนั้นต่อ 1 USD) ตั้งแต่ ม.ค. 2010"""
    rows, today = [], date.today()
    for y in range(2010, today.year + 1):
        end = min(date(y, 12, 31), today)
        js = get_json(f"https://api.frankfurter.dev/v1/{y}-01-01..{end}", {"base": "USD"})
        for d, rates in js["rates"].items():
            rows += [(d, cur, v) for cur, v in rates.items()]
        print("fxlong", y, len(js["rates"]), "days")
    df = pd.DataFrame(rows, columns=["date", "currency", "per_usd"])
    df["month"] = pd.to_datetime(df.date).dt.to_period("M").dt.to_timestamp()
    m = df.groupby(["month", "currency"], as_index=False).per_usd.mean()
    thb = m[m.currency == "THB"].set_index("month").per_usd.rename("thb_per_usd")
    m = m.join(thb, on="month")
    m["thb_per_unit"] = m.thb_per_usd / m.per_usd
    m.rename(columns={"month": "date"}).to_csv(OUT / "fx_monthly_usd_base_ecb_2010.csv", index=False, encoding="utf-8-sig")
    print("saved fxlong", m.shape)


def fetch_wb_long():
    frames = []
    for code, name in WB_INDICATORS.items():
        js = get_json(f"https://api.worldbank.org/v2/country/all/indicator/{code}",
                      {"format": "json", "date": "2008:2025", "per_page": 20000})
        df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), name: r["value"]} for r in js[1]])
        frames.append(df.set_index(["iso3", "year"]))
        print("wblong", code, len(df))
    meta = pd.read_csv(OUT / "worldbank_country_meta.csv", keep_default_na=False, na_values=[""])
    wide = pd.concat(frames, axis=1).reset_index().merge(meta[["iso3", "iso2", "name_en", "is_aggregate"]], on="iso3", how="left")
    wide = wide[~wide.is_aggregate.fillna(True).astype(bool)].drop(columns="is_aggregate")
    wide = wide[["iso2", "iso3", "name_en", "year", *WB_INDICATORS.values()]].sort_values(["iso3", "year"])
    wide.to_csv(OUT / "worldbank_indicators_yearly_2008.csv", index=False, encoding="utf-8-sig")
    print("saved wblong", wide.shape)


if __name__ == "__main__":
    steps = {"fx": fetch_fx, "wb": fetch_wb, "wx": fetch_wx, "fxlong": fetch_fx_long, "wblong": fetch_wb_long}
    for s in (sys.argv[1:] or ["fx", "wb", "wx"]):
        steps[s]()
