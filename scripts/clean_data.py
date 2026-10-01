"""Clean ข้อมูลดิบใน DATASET/ -> DATASET/clean/  (ไม่แก้ไฟล์ raw)

หลักการ: ไม่ลบข้อมูลที่ไม่แน่ใจ, แก้เฉพาะรูปแบบ (ชื่อคอลัมน์ ชนิดข้อมูล ช่องว่าง เดือน ปี)
และจัดการแถวซ้ำที่ยืนยันได้ว่าซ้ำจริง ทุกการตัดสินใจถูกนับและพิมพ์ลง stdout

ใช้: python scripts/clean_data.py
"""
import re
import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "DATASET"
OUT = RAW / "clean"
for sub in ("trade", "supply_side"):
    (OUT / sub).mkdir(parents=True, exist_ok=True)

LOG = {}
TH_MONTH_FULL = {m: i for i, m in enumerate(
    "มกราคม กุมภาพันธ์ มีนาคม เมษายน พฤษภาคม มิถุนายน กรกฎาคม สิงหาคม กันยายน ตุลาคม พฤศจิกายน ธันวาคม".split(), 1)}
TH_MONTH_FULL["กรกฏาคม"] = 7  # สะกดแบบ ฏ ที่พบในไฟล์ผลผลิตกุ้งรายเดือนปี 2567
TH_MONTH_ABBR = {m: i for i, m in enumerate(
    "ม.ค. ก.พ. มี.ค. เม.ย. พ.ค. มิ.ย. ก.ค. ส.ค. ก.ย. ต.ค. พ.ย. ธ.ค.".split(), 1)}


# ชื่อประเทศในไฟล์รายวันที่สะกดไม่ตรงกับ trade_hs (map มือ -> ISO2); ทะเลหลวง ไม่ใช่ประเทศ จึงไม่ map
COUNTRY_ALIASES = {
    "สาธารณรัฐเช็ก": "CZ", "ซาอุดีอาระเบีย": "SA", "มอลตา": "MT", "เรอูว์นียง": "RE", "สาธารณรัฐฮังการี": "HU",
    "บรูไนดารุสซาลาม": "BN", "ตูนิเซีย": "TN", "ยูกันดา": "UG", "ซูรินาเม": "SR", "สาธารณรัฐโกตดิวัวร์": "CI",
    "สาธารณรัฐเอลซัลวาดอร์": "SV", "เฟรนช์โปลินีเซีย": "PF", "นิการากัว": "NI", "สโลวีเนีย": "SI", "เบลีช": "BZ",
    "อเมริกันซามัว": "AS", "เวเนซุเอลา": "VE", "ไลบีเรีย": "LR", "คูราเซา": "CW", "ลักเซมเบิร์ก": "LU",
    "โดมินิกา": "DM", "สาธารณรัฐประชาธิปไตยคองโก": "CD", "เฟรนซืเกียนา": "GF", "ติมอร์ตะวันออก": "TL",
    "มาร์ตินีก": "MQ", "แอฟริกากลาง": "CF", "เคปเวิร์ด": "CV", "สาธารณรัฐเซาท์ซูดาน": "SS", "แซงบาร์เตเลมี": "BL",
    "คอโมโรส": "KM", "อิเควทอเรียลกินี": "GQ", "ชาด": "TD", "เนเธอร์แลนด์แอนทิลลิส": "AN", "คีร์กิซ": "KG",
}


def read_csv(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def squash(s):
    """strip + ยุบช่องว่างซ้อน; ค่าว่างเป็น NA"""
    s = s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.replace("", pd.NA)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def save(df, path, name):
    df.to_csv(OUT / path, index=False, encoding="utf-8-sig")
    LOG[name] = {"rows": len(df), "cols": df.shape[1]}
    print(f"saved {path}: {df.shape}")


# ------------------------------------------------------------------ trade_hs
def clean_trade_hs():
    raw = pd.concat([read_csv(f) for f in sorted((RAW / "trade").glob("dof_trade_hs_monthly_*.csv"))], ignore_index=True)
    n0 = len(raw)
    raw = raw.drop_duplicates()
    LOG["trade_hs_exact_dups_dropped"] = n0 - len(raw)
    df = pd.DataFrame({
        "year_be": num(raw.year).astype("int32"),
        "month": num(raw.month).astype("int8"),
        "hs11": raw.heading11.str.zfill(11),
        "country_iso2": squash(raw.countryID).str.upper(),
        "country_th": squash(raw.countryNameTH),
        "flow": raw.tradeflow.map({"1": "import", "2": "export"}),  # อนุมาน: ดู DICTIONARY
        "weight_kg": num(raw.weight),
        "quantity": num(raw.quantity),
        "value_thb": num(raw.price),
        "product_en": squash(raw.productDetailEN).replace({"(blank)": pd.NA}),
        "fish_name_th": squash(raw.fishName),
        "product_th": squash(raw.productDetailTH),
    })
    assert df.flow.notna().all()
    df.insert(0, "year", df.year_be - 543)
    df.insert(3, "hs4", df.hs11.str[:4])
    miss = df.country_iso2.isna()
    LOG["trade_hs_iso2_missing_in_raw"] = int(miss.sum())
    df["is_unknown_country"] = df.country_iso2.eq("YY")  # รหัส 'yy' = ประเทศสำหรับใช้นอกเขตต่อเนื่อง
    df.loc[df.is_unknown_country, "country_iso2"] = pd.NA
    df.insert(2, "date", pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.strftime("%Y-%m-%d"))
    key = ["year", "month", "hs11", "country_iso2", "country_th", "flow"]
    df["is_key_dup"] = df.duplicated(key, keep=False)
    LOG["trade_hs_rows_sharing_key"] = int(df.is_key_dup.sum())
    df = df.sort_values(key).reset_index(drop=True)
    save(df, "trade/trade_hs_monthly.csv", "trade_hs_monthly")

    # panel: รวมแถวที่ key ซ้ำ -> 1 แถวต่อ เดือน x HS11 x ประเทศ x ทิศทาง (สำหรับ modeling)
    g = (df.groupby(["year", "month", "date", "hs11", "hs4", "country_iso2", "country_th", "flow"], dropna=False, as_index=False)
           [["weight_kg", "quantity", "value_thb"]].sum())
    save(g, "trade/trade_hs_monthly_panel.csv", "trade_hs_monthly_panel")
    assert abs(g.value_thb.sum() - df.value_thb.sum()) < 1
    return df


# --------------------------------------------------------------- daily trade
DAILY_COLS = ["date", "day", "month", "year", "checkpoint", "country_th", "transport", "name_th", "name_common",
              "name_sci", "group", "water", "form", "qty_reported", "unit_reported", "qty_kg", "unit_std", "value_thb"]


def clean_daily(flow, country_map):
    files = sorted((RAW / "trade").glob(f"dof_{flow}_daily_*"))
    frames = []
    for f in files:
        d = pd.read_excel(f, dtype=str)  # .xlsx / .xls; header ชื่อต่างกันตามปี -> ใช้ตำแหน่ง
        assert d.shape[1] == 18, (f, d.shape)
        d.columns = DAILY_COLS
        d["source_file"] = f.name
        frames.append(d)
        print(" read", f.name, d.shape)
    raw = pd.concat(frames, ignore_index=True)
    n_dup = int(raw.drop(columns="source_file").duplicated().sum())
    df = pd.DataFrame({
        "date": pd.to_datetime(raw.date, errors="raise"),
        "checkpoint": squash(raw.checkpoint),
        "country_th": squash(raw.country_th),
        "transport": squash(raw.transport),
        "name_th": squash(raw.name_th),
        "name_common": squash(raw.name_common),
        "name_sci": squash(raw.name_sci),
        "group": squash(raw.group),
        "water": squash(raw.water),
        "form": squash(raw.form),
        "qty_reported": num(raw.qty_reported),
        "unit_reported": squash(raw.unit_reported),
        "qty_kg": num(raw.qty_kg),
        "value_thb": num(raw.value_thb),
        "source_file": raw.source_file,
    })
    assert (raw.unit_std.dropna().str.strip() == "KGM").all()  # unit_std เป็น KGM ทั้งหมด จึงตัดคอลัมน์ทิ้ง
    # ตรวจว่าคอลัมน์วัน/เดือน/ปี สอดคล้องกับ date
    assert (df.date.dt.year == num(raw.year)).all() and (df.date.dt.month == num(raw.month)).all()
    df["country_iso2"] = df.country_th.map(country_map)
    df["is_dup_row"] = df.drop(columns="source_file").duplicated(keep=False)
    LOG[f"{flow}_daily_exact_dup_rows"] = n_dup
    LOG[f"{flow}_daily_country_iso_coverage"] = round(float(df.country_iso2.notna().mean()), 4)
    LOG[f"{flow}_daily_zero_value_rows"] = int((df.value_thb == 0).sum())
    df.insert(1, "year", df.date.dt.year.astype("int16"))
    df.insert(2, "month", df.date.dt.month.astype("int8"))
    df["date"] = df.date.dt.strftime("%Y-%m-%d")
    df = df.sort_values(["date", "checkpoint", "country_th", "name_th"]).reset_index(drop=True)
    save(df, f"trade/trade_daily_{flow}.csv", f"{flow}_daily")


# --------------------------------------------------------------- health cert
def clean_health_cert():
    frames = []
    for f in sorted((RAW / "trade").glob("dof_health_cert_by_country_*.csv")):
        d = read_csv(f).rename(columns={"จำนวน": "count", "ปริมาณ": "count"})  # ชื่อคอลัมน์ไม่ตรงกันข้ามปี
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    name = raw["รายชื่อประเทศ"].str.strip()
    iso = name.str.extract(r"_\[([A-Z]{2})\]$")[0]
    df = pd.DataFrame({
        "year_be": num(raw["ปี"]).astype("int32"),
        "month": raw["เดือน"].str.strip().map(TH_MONTH_ABBR),
        "country_iso2": iso.fillna(name.where(name.eq("ประเทศอื่นๆ"), pd.NA).map({"ประเทศอื่นๆ": "OTHER"})),
        "country_name": name.str.replace(r"\s*_\[[A-Z]{2}\]$", "", regex=True).str.strip().str.title().replace({"ประเทศอื่นๆ": "ประเทศอื่นๆ"}),
        "cert_count": num(raw["count"]),
    })
    assert df.month.notna().all() and df.country_iso2.notna().all() and df.cert_count.notna().all()
    df["month"] = df.month.astype("int8")
    df["year"] = df.year_be - 543
    # GERMANY ม.ค. 2568 ซ้ำ 2 แถว (0 และ 19) แยกไม่ออกว่าอันไหนถูก -> รวมกัน (=19) และบันทึกไว้
    dup = df.duplicated(["year_be", "month", "country_iso2"], keep=False)
    LOG["health_cert_key_dup_rows_merged"] = int(dup.sum())
    df = df.groupby(["year", "year_be", "month", "country_iso2", "country_name"], as_index=False).cert_count.sum()
    df.insert(0, "date", pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.strftime("%Y-%m-%d"))
    df = df.sort_values(["date", "country_iso2"]).reset_index(drop=True)
    save(df, "trade/health_cert_by_country_monthly.csv", "health_cert")


# ------------------------------------------------------------- species catalog
def clean_catalog():
    raw = read_csv(RAW / "trade" / "dof_species_hs_catalog.csv")
    cust = raw["พิกัดศุลกากร"].str.strip()
    df = pd.DataFrame({
        "item_code": squash(raw["รหัส"]),
        "hs11": cust.str.extract(r"^([\d.\-]+)/")[0].str.replace(r"\D", "", regex=True),
        "customs_unit_code": cust.str.extract(r"/([A-Z0-9]+)\s*-")[0],
        "product_main": squash(raw["สินค้าหลัก"]),
        "product_sub": squash(raw["สินค้ารอง"]),
        "name_th": squash(raw["ชื่อไทย"]),
        "name_common": squash(raw["ชื่อสามัญ"]),
        "genus": squash(raw["Genius"]),  # ต้นฉบับสะกด Genius
        "species": squash(raw["Species"]),
        "unit_code": squash(raw["หน่วยนับ"]).str.extract(r"^([A-Z0-9]+)")[0],
    })
    LOG["catalog_hs11_not_11_digits"] = int((df.hs11.str.len() != 11).sum())
    LOG["catalog_item_code_dups"] = int(df.item_code.duplicated().sum())
    df["hs4"] = df.hs11.str[:4]
    save(df, "trade/species_hs_catalog.csv", "catalog")


# ---------------------------------------------------------------- supply side
def clean_production():
    for kind, unit in (("quantity", "q"), ("value", "v")):
        src = read_csv(RAW / "supply_side" / f"dof_production_by_source_{kind}.csv")
        a = src.set_axis(["year_be", "total", "marine_capture", "freshwater_capture", "coastal_aquaculture",
                          "freshwater_aquaculture"], axis=1).apply(num)
        a["year"] = a.year_be - 543
        save(a, f"supply_side/production_by_source_{kind}.csv", f"prod_source_{kind}")
        grp = read_csv(RAW / "supply_side" / f"dof_production_by_species_group_{kind}.csv")
        b = grp.set_axis(["year_be", "total", "marine_fish", "marine_shrimp", "marine_crab", "marine_squid",
                          "marine_shellfish", "marine_other", "freshwater_fish", "freshwater_shrimp",
                          "freshwater_other"], axis=1).apply(num)
        b["year"] = b.year_be - 543
        save(b, f"supply_side/production_by_species_group_{kind}.csv", f"prod_group_{kind}")


def clean_marine():
    frames = []
    for f in sorted((RAW / "supply_side").glob("dof_marine_catch_monthly_*")):
        d = pd.read_excel(f, dtype=str) if f.suffix == ".xlsx" else read_csv(f)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    df = pd.DataFrame({
        "year_be": num(raw["ปี"]).astype("int32"),
        "month": squash(raw["เดือน"]).map(TH_MONTH_FULL),
        "fishing_type": squash(raw["ประเภทการทำการประมง"]),
        "gear": squash(raw["เครื่องมือ"]),
        "vessel_size": squash(raw["ขนาดเรือ"]),
        "fishing_area": squash(raw["พื้นที่ทำการประมง"]),
        "species_th": squash(raw["ชนิดสัตว์น้ำ"]),
        "quantity": num(raw["ปริมาณ"]),
        "value": num(raw["มูลค่า"]),
    })
    assert df.month.notna().all() and df.quantity.notna().all() and df.value.notna().all()
    df["month"] = df.month.astype("int8")
    df.insert(0, "year", df.year_be - 543)
    n0 = len(df)
    LOG["marine_exact_dups"] = int(df.duplicated().sum())
    LOG["marine_rows"] = n0
    save(df, "supply_side/marine_catch_monthly.csv", "marine_catch")


def clean_aquaculture():
    parts = {"freshwater": "freshwater", "brackish_fish": "brackish_fish", "marine_shrimp": "marine_shrimp",
             "marine_crab": "marine_crab", "marine_shellfish": "marine_shellfish"}
    frames = []
    for key in parts:
        f = next((RAW / "supply_side").glob(f"dof_aquaculture_{key}_*.csv"))
        d = read_csv(f)
        frames.append(pd.DataFrame({
            "category": key,
            "year_be": num(d["ปี"]).astype("int32"),
            "culture_type": squash(d["ประเภทการเพาะเลี้ยง"]),
            "province": squash(d["จังหวัด"]),
            "method": squash(d["ประเภทการเลี้ยง"]) if "ประเภทการเลี้ยง" in d else pd.NA,
            "species_th": squash(d["ชนิดสัตว์น้ำ"]) if "ชนิดสัตว์น้ำ" in d else pd.NA,
            "farms": num(d["จำนวนฟาร์ม"]),
            "area": num(d["เนื้อที่เลี้ยง"]),
            "quantity": num(d["ปริมาณ"]),
            "value": num(d["มูลค่า"]),
        }))
    df = pd.concat(frames, ignore_index=True)
    df.insert(1, "year", df.year_be - 543)
    LOG["aquaculture_null_numeric"] = df[["farms", "area", "quantity", "value"]].isna().sum().to_dict()
    LOG["aquaculture_key_dups"] = int(df.duplicated(["category", "year", "province", "method", "species_th"], keep=False).sum())
    save(df, "supply_side/aquaculture_yearly.csv", "aquaculture")


def clean_nabc_json():
    js = json.loads((RAW / "supply_side" / "nabc_shrimp_production_annual_2565_2567.json").read_text(encoding="utf-8"))
    d = pd.DataFrame(js).rename(columns={"year": "year_be", "subAttribute": "sub_attribute", "values": "value"})
    d.insert(1, "year", d.year_be - 543)
    save(d, "supply_side/shrimp_production_annual.csv", "nabc_shrimp")


def clean_shrimp_monthly():
    """ผลผลิตกุ้งทะเลรายเดือน x จังหวัด x ชนิด (กรมประมง afpd-appd) รวม 3 ไฟล์

    ไฟล์ที่ catalog ตั้งชื่อ "ปี 2569" ไม่มีคอลัมน์ปี, ม.ค.-เม.ย. ตรงกับไฟล์ปี 2568 ทุกแถว และมีเดือนถึง ธ.ค. (ซึ่งยังมาไม่ถึงในปี 2569)
    -> ตีความว่าเป็นข้อมูลปี 2568 เต็มปี (year_inferred = True) ใช้ไฟล์นี้เป็นตัวแทนปี 2568 และทิ้งไฟล์ปี 2568 ที่เป็นเซตย่อย
    """
    sup = RAW / "supply_side"
    a = read_csv(sup / "dof_shrimp_production_monthly_2567.csv")
    b = read_csv(sup / "dof_shrimp_production_monthly_2568.csv")
    c = read_csv(sup / "dof_shrimp_production_monthly_2569.csv")
    c = c.rename(columns={"Attribute": "ชนิด", "Value": "จำนวน"}).assign(ปี="2568")
    key = ["จังหวัด", "เดือน", "ชนิด"]
    ov = b.merge(c, on=key, suffixes=("_b", "_c"))
    assert len(ov) == len(b) and (ov["จำนวน_b"] == ov["จำนวน_c"]).all()  # ไฟล์ 2568 เป็นเซตย่อยที่เหมือนกันทุกแถว
    LOG["shrimp_monthly_2568_file_subset_of_2569_file"] = True
    parts = [a.assign(year_inferred=False, source_file="dof_shrimp_production_monthly_2567.csv"),
             c.assign(year_inferred=True, source_file="dof_shrimp_production_monthly_2569.csv")]
    raw = pd.concat(parts, ignore_index=True)
    df = pd.DataFrame({
        "year_be": num(raw["ปี"]).astype("int32"),
        "month": squash(raw["เดือน"]).map(TH_MONTH_FULL),
        "province": squash(raw["จังหวัด"]),
        "region": squash(raw["ภาค"]),
        "species": squash(raw["ชนิด"]),
        "quantity": num(raw["จำนวน"]),  # หน่วยไม่ระบุ (อนุมาน: ตัน)
        "year_inferred": raw.year_inferred,
        "source_file": raw.source_file,
    })
    assert df.month.notna().all() and df.quantity.notna().all()
    df["month"] = df.month.astype("int8")
    df.insert(0, "year", df.year_be - 543)
    df.insert(0, "date", pd.to_datetime(dict(year=df.year, month=df.month, day=1)).dt.strftime("%Y-%m-%d"))
    LOG["shrimp_monthly_key_dups"] = int(df.duplicated(["year", "month", "province", "species"]).sum())
    LOG["shrimp_monthly_months"] = int(df[["year", "month"]].drop_duplicates().shape[0])
    df = df.sort_values(["date", "province", "species"]).reset_index(drop=True)
    save(df, "supply_side/shrimp_production_monthly_province.csv", "shrimp_monthly")


def clean_biomass():
    raw = read_csv(RAW / "supply_side" / "dof_biomass_2563_2567.csv")
    parse = lambda s: pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    long = pd.concat([
        pd.DataFrame({"year_be": num(raw["ปี"]), "scientific_name": squash(raw["Scientific name"]), "area": area, "biomass": parse(raw[col])})
        for col, area in (("อ่าวไทย", "gulf_of_thailand"), ("อันดามัน", "andaman"))
    ], ignore_index=True).dropna(subset=["biomass"])  # ว่าง = ไม่มีการสำรวจ/ไม่พบในพื้นที่นั้น
    long["year_be"] = long.year_be.astype("int32")
    long.insert(0, "year", long.year_be - 543)
    save(long.sort_values(["year", "scientific_name", "area"]), "supply_side/biomass_survey_yearly.csv", "biomass")


# ---------------------------------------------------------------- dim country
def build_dim_country(trade):
    wb = pd.read_csv(RAW / "external" / "worldbank_country_meta.csv", keep_default_na=False, na_values=[""])  # iso2 "NA" = นามิเบีย
    wb = wb[~wb.is_aggregate][["iso2", "iso3", "name_en", "region", "income_level"]]
    c = (trade.dropna(subset=["country_iso2"]).groupby("country_iso2", as_index=False)
         .agg(country_th=("country_th", lambda s: s.mode().iat[0])))
    extra = pd.DataFrame({"country_iso2": list(COUNTRY_ALIASES.values()), "country_th": list(COUNTRY_ALIASES)})
    c = pd.concat([c, extra[~extra.country_iso2.isin(c.country_iso2)]], ignore_index=True)
    c = c.merge(wb, left_on="country_iso2", right_on="iso2", how="left").drop(columns="iso2")
    c["in_worldbank"] = c.iso3.notna()
    LOG["dim_country_namibia_ok"] = bool(c.loc[c.country_iso2 == "NA", "in_worldbank"].all())
    LOG["dim_country_not_in_worldbank"] = c.loc[~c.in_worldbank, "country_iso2"].tolist()
    save(c, "trade/dim_country.csv", "dim_country")
    return c


if __name__ == "__main__":
    t = clean_trade_hs()
    dim = build_dim_country(t)
    cmap = {**COUNTRY_ALIASES, **dict(zip(t.dropna(subset=["country_iso2"]).country_th, t.dropna(subset=["country_iso2"]).country_iso2))}
    LOG["trade_hs_iso2_still_missing_not_yy"] = int((t.country_iso2.isna() & ~t.is_unknown_country).sum())
    clean_health_cert()
    clean_catalog()
    clean_production()
    clean_marine()
    clean_aquaculture()
    clean_nabc_json()
    clean_shrimp_monthly()
    clean_biomass()
    for flow in ("export", "import"):
        clean_daily(flow, cmap)
    (OUT / "cleaning_log.json").write_text(json.dumps(LOG, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(LOG, ensure_ascii=False, indent=1, default=str))
