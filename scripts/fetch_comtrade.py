"""ดึงการค้าสินค้าประมงของไทยรายเดือน x HS4 x ประเทศคู่ค้า จาก UN Comtrade (public preview endpoint, ไม่ต้องใช้ key)

ข้อจำกัดของ preview endpoint ที่ตรวจแล้ว: 1 request = 1 เดือน x 1 HS x 1 ทิศทาง (ส่งหลายเดือน/หลาย HS ไม่ได้)
และมี **โควตาจำนวนครั้งที่เรียก** (เกินแล้วตอบ 403 "Out of call volume quota. Quota will be replenished in HH:MM:SS")
-> script รันทีละ request, ถ้าโควตาหมดจะนอนรอจนโควตาคืนแล้วทำต่ออัตโนมัติ, cache ทุก request (Ctrl+C แล้วรันใหม่ต่อได้)
ลำดับ: HS ที่สำคัญก่อน (กุ้ง 0306 -> ปลาปรุงแต่ง 1604 -> ปลาแช่แข็ง 0303 ...) ทำให้ถ้าหยุดกลางทางก็ได้อนุกรมที่ครบของ HS ต้น ๆ

ใช้: python scripts/fetch_comtrade.py X          (เฉพาะ export)
     python scripts/fetch_comtrade.py build      (สร้าง CSV จาก cache เท่าที่มี ไม่ยิง API)
Output: DATASET/external/comtrade_thailand_seafood_monthly.csv, cache ใน DATASET/external/_comtrade_cache.jsonl
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

EXT = Path(__file__).resolve().parent.parent / "DATASET" / "external"
CACHE = EXT / "_comtrade_cache.jsonl"
OUT = EXT / "comtrade_thailand_seafood_monthly.csv"
HS4 = ["0306", "1604", "0303", "0307", "0304", "0302", "0305", "0301", "1605", "0308"]  # เรียงตามความสำคัญ
START, END = (2010, 1), (2026, 4)  # ตรวจแล้ว: ก่อน ม.ค. 2010 บางปีว่าง, หลัง เม.ย. 2026 ยังไม่มี
URL = ("https://comtradeapi.un.org/public/v1/preview/C/M/HS?reporterCode=764&partnerCode=&partner2Code=0"
       "&customsCode=C00&motCode=0&flowCode={flow}&period={period}&cmdCode={hs}")
REF = "https://comtradeapi.un.org/files/v1/app/reference/partnerAreas.json"
SLEEP = 1.5  # วินาทีระหว่าง request


def http(url, tries=6):
    fails = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "ignore")
            if e.code == 403 and "quota" in body.lower():
                m = re.search(r"(\d+):(\d+):(\d+)", body)
                wait = (int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])) + 30 if m else 1800
                print(f"  โควตาหมด รอ {wait // 60} นาที แล้วทำต่อ ({time.strftime('%H:%M:%S')})", flush=True)
                time.sleep(wait)
                continue  # ไม่นับเป็นความล้มเหลว
            if e.code == 429:
                time.sleep(60)
                continue
            fails += 1
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            # เน็ตหลุด / DNS หาไม่เจอ / timeout: ไม่ใช่ความผิดของ request นี้ -> รอเน็ตกลับมาแล้วลองใหม่ ไม่นับว่าล้มเหลว
            print(f"  เชื่อมต่อไม่ได้ ({type(e).__name__}) รอ 30 วินาทีแล้วลองใหม่ ({time.strftime('%H:%M:%S')})", flush=True)
            time.sleep(30)
            continue
        except Exception as e:  # noqa: BLE001
            fails += 1
            err = repr(e)
        if fails >= tries:
            raise RuntimeError(f"{err}: {url}")
        time.sleep(5 * fails)


def months():
    y, m = START
    while (y, m) <= END:
        yield f"{y}{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def load_done():
    done = set()
    if CACHE.exists():
        for line in CACHE.open(encoding="utf-8"):
            d = json.loads(line)
            done.add((d["flow"], d["period"], d["hs"]))
    return done


def work(task):
    flow, period, hs = task
    js = http(URL.format(flow=flow, period=period, hs=hs))
    rows = js.get("data") or []
    assert len(rows) < 500, f"hit 500-row cap {task}"  # ถ้าชน cap ต้องแยกคำขอ
    keep = ["period", "flowCode", "cmdCode", "partnerCode", "primaryValue", "cifvalue", "fobvalue", "netWgt", "qty", "qtyUnitAbbr"]
    rec = {"flow": flow, "period": period, "hs": hs, "rows": [{k: r.get(k) for k in keep} for r in rows]}
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(rows)


def build():
    recs = [json.loads(l) for l in CACHE.open(encoding="utf-8")]
    rows = [dict(r, flow=x["flow"], hs4=x["hs"]) for x in recs for r in x["rows"]]
    df = pd.DataFrame(rows)
    ref = http(REF)["results"]
    p = pd.DataFrame(ref)[["PartnerCode", "PartnerDesc", "PartnerCodeIsoAlpha2", "PartnerCodeIsoAlpha3"]]
    p.columns = ["partnerCode", "partner_name", "partner_iso2", "partner_iso3"]
    df = df.merge(p, on="partnerCode", how="left")
    df["year"] = df.period.str[:4].astype(int)
    df["month"] = df.period.str[4:].astype(int)
    df["flow"] = df.flow.map({"X": "export", "M": "import"})
    df = df.rename(columns={"primaryValue": "value_usd", "netWgt": "net_weight_kg", "qty": "qty", "qtyUnitAbbr": "qty_unit",
                            "cifvalue": "cif_usd", "fobvalue": "fob_usd"})
    df["is_world_total"] = df.partnerCode == 0
    cols = ["year", "month", "flow", "hs4", "partnerCode", "partner_iso2", "partner_iso3", "partner_name", "is_world_total",
            "value_usd", "net_weight_kg", "qty", "qty_unit", "cif_usd", "fob_usd"]
    df = df[cols].sort_values(["flow", "hs4", "year", "month", "partnerCode"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved", OUT, df.shape)


if __name__ == "__main__":
    args = sys.argv[1:] or ["X"]
    if args != ["build"]:
        done = load_done()
        for flow in args:
            tasks = [(flow, p, h) for h in HS4 for p in months() if (flow, p, h) not in done]
            print(f"flow {flow}: {len(tasks)} requests to go", flush=True)
            t0, failed = time.time(), []
            for i, t in enumerate(tasks, 1):
                try:
                    work(t)
                except RuntimeError as e:  # ข้ามตัวที่พัง รันใหม่จะลองซ้ำเอง
                    failed.append(t)
                    print("  skip", e, flush=True)
                time.sleep(SLEEP)
                if i % 50 == 0:
                    print(f"  {i}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
            print(f"flow {flow} done, skipped {len(failed)} (รันซ้ำเพื่อลองใหม่)", flush=True)
    build()
