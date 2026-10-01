# Thai Fisheries Trade — ML Project (CRISP-DM)

โปรเจกต์วิชา ML: ใช้ Open Data ของไทยอย่างน้อย 1 ชุด วิเคราะห์และสร้างโมเดลตามกระบวนการ CRISP-DM และดึงข้อมูลภายนอก (สภาพอากาศ, เศรษฐกิจ, API อื่น ๆ) มาเพิ่มมิติของ feature ได้

> **สถานะ:** ยังอยู่ขั้นเลือกโจทย์และสำรวจข้อมูล (Business Understanding / Data Understanding)

---

> **Notebook หลัก: [`main.ipynb`](main.ipynb)** — เดินตาม CRISP-DM มี Business Understanding, Data Understanding และ Data Preparation แล้ว (พร้อมผลลัพธ์และกราฟ) ขั้นตอนถัดไป (Modeling) จะเพิ่มต่อในไฟล์เดียวกัน ใช้ `pip install pandas numpy matplotlib`
> ถ้าต้องการแชร์เฉพาะส่วน Business + Data Understanding: เปิด [`reports/main_BU_DU.html`](reports/main_BU_DU.html) (ภาพนิ่ง ไม่ต้องใช้ Jupyter) · ข้อมูลที่เตรียมแล้วอยู่ที่ `DATASET/prepared/`
> **โจทย์หลัก = พยากรณ์ยอดส่งออกรายเดือน** (ตามแผนเดิม) ตั้งความคาดหวังล่วงหน้าจากข้อมูลว่า ML อาจไม่ชนะ baseline และรายงานผลตามจริง ส่วน Clustering ตลาดเป็นอีกงาน ML ที่ต้องส่งมอบ (และช่วยอธิบายความแม่น) และ Early Warning เป็นส่วนเสริม รายละเอียดอยู่ใน `main.ipynb` ส่วน 1.4 และ 2.2.1

---

## 1. ข้อมูลตั้งต้น

| รายการ | รายละเอียด |
|---|---|
| ชื่อชุดข้อมูล | ปริมาณและมูลค่าการนำเข้าส่งออกสินค้าประมง (`impexp_product`) |
| หน่วยงาน | กรมประมง |
| ลิงก์ | [catalog.fisheries.go.th/dataset/impexp_product](https://catalog.fisheries.go.th/dataset/impexp_product/resource/53896d05-43eb-4f7c-916e-fd7893d7970b) · [GD Catalog mirror](https://gdcatalog.go.th/dataset/gdpublish-impexp-product) |

**ข้อจำกัด:** ชุดนี้น่าจะเป็นตัวเลขรวมระดับประเทศ (รายปีหรือรายเดือน) จึงมีจำนวนแถวน้อยเกินไปสำหรับเทรนโมเดล ML
→ ใช้ชุดนี้เป็น **baseline และตัวเช็กความถูกต้อง (sanity check)** ส่วนข้อมูลหลักสำหรับเทรนมาจาก API ของกระทรวงพาณิชย์ (หัวข้อ 2.1)

---

## 2. ชุดข้อมูลที่เกี่ยวข้อง (ข้อมูลจริงจากหน่วยงานรัฐ)

ความหมายของสถานะ:
- ✔ = เปิดหน้า metadata และอ่านรายละเอียดแล้ว
- ◐ = เห็นในผลค้นหาว่ามีอยู่บนพอร์ทัลรัฐจริง แต่ยังไม่ได้เปิดดูคอลัมน์ **ต้องเช็กเองก่อนใช้**

### 2.1 การค้าสินค้าประมง

| ชุดข้อมูล | หน่วยงาน | สถานะ | บทบาทในโปรเจกต์ |
|---|---|---|---|
| [ข้อมูลตลาดส่งออกรายพิกัดศุลกากร (API)](https://data.moc.go.th/OpenData/ExportHarmonizeCountries) | กระทรวงพาณิชย์ | ✔ | **ข้อมูลหลัก**: รายเดือน × ประเทศ × HS code มีปริมาณและมูลค่า (USD/THB) |
| [ข้อมูลสินค้าส่งออก-นำเข้า](https://data.go.th/dataset/dataset_31_01) | กระทรวงพาณิชย์ | ✔ | มาจากข้อมูลศุลกากร รายเดือน ตั้งแต่ปี 1991 สัญญาอนุญาต CC-BY |
| [API ค้นหารหัสสินค้า](https://data.moc.go.th/OpenData/Products) | กระทรวงพาณิชย์ | ✔ | หา HS code จากชื่อสินค้า |
| [สถิติการนำเข้าสัตว์น้ำ](https://gdcatalog.go.th/dataset/gdpublish-import-pro) | กรมประมง | ◐ | ฝั่งนำเข้า |
| [สถิติการส่งออกสัตว์น้ำ](https://gdcatalog.go.th/dataset/gdpublish-export-pro) | กรมประมง | ◐ | ฝั่งส่งออก ใช้เทียบกับข้อมูลกระทรวงพาณิชย์ |
| [พิกัดสินค้าประมง](https://catalog.fisheries.go.th/en/dataset/datahamonize) | กรมประมง | ◐ | lookup table สำหรับ map HS code ↔ ชื่อสินค้าประมง |
| [หนังสือรับรองสุขภาพสัตว์น้ำเพื่อการส่งออก](https://catalog.fisheries.go.th/en/dataset/aahrdd-healthcer/resource/f0e4234d-dcf8-4b25-95c6-3924f7d374ce?inner_span=True) | กรมประมง | ◐ | leading indicator ของการส่งออก |

### 2.2 ฝั่งผลผลิต (supply-side features)

| ชุดข้อมูล | หน่วยงาน | สถานะ |
|---|---|---|
| [ข้อมูลผลผลิตกุ้งทะเลรายเดือน](https://nabc-catalog.oae.go.th/dataset/fisheries_production_shrimp) | กรมประมง (ผ่าน NABC) | ◐ → ใช้ชุดต้นทาง [afpd-appd](https://catalog.fisheries.go.th/dataset/afpd-appd) ดาวน์โหลดแล้ว (23 เดือน ดู DICTIONARY 3.5) |
| [ปริมาณและมูลค่าสัตว์น้ำทั้งหมด](https://gdcatalog.go.th/dataset/gdpublish-dofd07-05-0101-02) | กรมประมง | ◐ |
| [ปริมาณและมูลค่าการจับสัตว์น้ำ](https://gdcatalog.go.th/dataset/gdpublish-dofd07-05-0101-03) | กรมประมง | ◐ |
| [ปริมาณและมูลค่าผลผลิตสัตว์น้ำจากการเพาะเลี้ยง](https://gdcatalog.go.th/dataset/gdpublish-dofd07-05-0101-04) | กรมประมง | ◐ |
| [สถิติการประมง](https://opendata.nesdc.go.th/dataset/fisheries-statistics) | กรมประมง (ผ่านพอร์ทัลของสภาพัฒน์) | ✔ ย้อนหลังประมาณ 10 ปี ส่วนใหญ่เป็น PDF |

### 2.3 ราคา (ใช้ได้จำกัด)

| ชุดข้อมูล | สถานะ | ข้อจำกัด |
|---|---|---|
| [ราคาสัตว์น้ำประจำวัน สะพานปลากรุงเทพ](https://data.go.th/en/dataset/item_55f5ccbd-f9c5-49f7-ba17-dc935cd7f5b7) | ✔ | หยุดอัปเดตตั้งแต่ ต.ค. 2021 |
| [ราคากุ้งขาวแวนนาไมที่เกษตรกรขายได้ (สศก.)](https://catalog.oae.go.th/dataset/data-baer-0406) | ◐ | ตอนเปิดจริงเด้งไปหน้า login |

### 2.4 ชุดที่หาเจอแต่ไม่แนะนำ
- [ข้อมูลสัตว์น้ำของไทย](https://data.go.th/dataset/item_cb5d392d-2824-4ac4-a171-2c4796ce81c2): เป็นข้อมูลอนุกรมวิธาน (taxonomy) ไม่มีตัวเลขการค้าหรือผลผลิต
- ระบบสถิติของกรมศุลกากรและ OAE impexp: เป็นระบบให้กดค้นทีละเงื่อนไข ไม่ใช่ open dataset ที่มีสัญญาอนุญาตระบุชัด

---

## 3. โจทย์ที่เป็นไปได้

### โจทย์หลัก (แนะนำ): พยากรณ์มูลค่าส่งออกสินค้าประมงรายเดือน แยกตามประเทศปลายทาง

**คำถาม:** เดือนหน้าไทยจะส่งออกกุ้ง ทูน่า หรือสินค้าประมงกลุ่มอื่นไปแต่ละประเทศได้มูลค่าเท่าไร

| หัวข้อ | รายละเอียด |
|---|---|
| ประเภทงาน | Regression / time-series forecasting บน panel data |
| หน่วยข้อมูล (1 แถว) | เดือน × ประเทศปลายทาง × HS code |
| Target | `value_baht` ของเดือน t+1 |
| Baseline | Seasonal naive, SARIMA |
| โมเดลหลัก | LightGBM / XGBoost ที่ใช้ lag features |
| Validation | Time-based split / rolling-origin **ห้าม random split** |
| Metric | MAE, MAPE (ระวังค่าใกล้ 0), RMSE |

**ใครได้ประโยชน์**
- โรงงานแปรรูปและผู้ส่งออก: วางแผนซื้อวัตถุดิบ กำลังการผลิต และสต็อก
- กรมส่งเสริมการค้าระหว่างประเทศ / กรมประมง: เห็นล่วงหน้าว่าตลาดไหนกำลังชะลอ จะได้ปรับนโยบายหรือจัดกิจกรรมส่งเสริมตลาดได้ทัน

### โจทย์เสริม A: Early warning ว่าตลาดไหนยอดส่งออกจะตกแรง
- Target: binary ว่ายอดลดลงเกิน X% YoY หรือไม่
- โมเดล: Logistic Regression เทียบกับ Random Forest ต้องจัดการ class imbalance
- ประโยชน์: ผู้ส่งออกกระจายความเสี่ยงไปตลาดอื่นได้ทัน

### โจทย์เสริม B: Clustering ตลาดส่งออก (ใช้ในขั้น EDA)
- Features: growth rate, volatility, unit value (บาท/กก.), product mix
- โมเดล: K-Means / DBSCAN แล้วตั้งชื่อ segment เช่น "ตลาดพรีเมียมโตช้า"
- ประโยชน์: ใช้วางกลยุทธ์ว่าควรบุกตลาดไหนด้วยสินค้าแบบไหน

### โจทย์ที่ความเสี่ยงสูง: พยากรณ์ผลผลิตหรือราคากุ้งจากสภาพอากาศ
- ข้อมูลราคายังเข้าถึงไม่ได้แน่นอน และจำนวนแถวน้อย จึงไม่แนะนำเป็นโจทย์หลัก

---

## 4. ข้อมูลภายนอกที่จะดึงมาเพิ่ม

| Feature | แหล่ง | Join key | เหตุผล |
|---|---|---|---|
| อัตราแลกเปลี่ยน THB/USD, JPY, CNY, EUR | **ดึงแล้ว** `external/fx_daily_usd_base_ecb.csv` (ECB ผ่าน Frankfurter; BOT ต้องมี token) | `date` (เดือน) + สกุลเงินของประเทศปลายทาง | บาทแข็งแล้วราคาสินค้าส่งออกแพงขึ้น |
| GDP, CPI, ประชากรของประเทศปลายทาง | **ดึงแล้ว** `external/worldbank_indicators_yearly.csv` (World Bank API) | `country_code` + ปี | กำลังซื้อของตลาด |
| ผลผลิตกุ้งทะเลรายเดือน | กรมประมง | `date` + กลุ่ม HS (กุ้ง) | ฝั่ง supply |
| สภาพอากาศของจังหวัดที่เลี้ยงกุ้ง/ท่าเรือประมง | **ดึงแล้ว** `external/weather_daily_shrimp_provinces.csv` (Open-Meteo) | `date` | กระทบผลผลิต (ใช้ในโจทย์ผลผลิต/ราคาเป็นหลัก) |
| หนังสือรับรองสุขภาพสัตว์น้ำ | กรมประมง | `date` | leading indicator |

**Features ที่สร้างเองจากข้อมูลหลัก**
- `lag_1`, `lag_3`, `lag_12`, rolling mean/std
- YoY growth
- `unit_value_baht = value_baht / quantity`
- ตัวแปร month/quarter (seasonality)
- ส่วนแบ่งตลาดของแต่ละประเทศ

> ข้อควรระวัง: ต้องไม่ให้ feature ใช้ข้อมูลของเดือนที่กำลังจะพยากรณ์ (data leakage) และต้องตรวจ `country_code` ของกระทรวงพาณิชย์ว่าเป็น ISO แบบเดียวกับที่ World Bank ใช้หรือไม่

---

## 5. โครงสร้างข้อมูลใน `DATASET/` (ดาวน์โหลดแล้ว)

รายละเอียดทุกไฟล์ (คอลัมน์, หน่วย, ช่วงเวลา, ปัญหาที่พบ, ตารางชื่อเดิม→ชื่อใหม่) อยู่ที่ **[DATASET/DICTIONARY.md](DATASET/DICTIONARY.md)**

| โฟลเดอร์ | เนื้อหา |
|---|---|
| `DATASET/trade/` | การค้าประมงของกรมประมง: รายเดือนตาม HS (`dof_trade_hs_monthly_*`), รายวันตามสายพันธุ์ (`dof_export_daily_*`, `dof_import_daily_*`), ใบรับรองสุขภาพ (`dof_health_cert_by_country_*`), lookup (`dof_species_hs_catalog`) |
| `DATASET/supply_side/` | ผลผลิตจับ/เพาะเลี้ยง รายปีและรายเดือน |
| `DATASET/price/` | มีแต่ metadata ของราคาสะพานปลา ยังไม่มีตัวเลขราคา |
| `DATASET/external/` | อัตราแลกเปลี่ยน (ECB), เศรษฐกิจรายประเทศ (World Bank), สภาพอากาศรายจังหวัด (Open-Meteo) ดึงด้วย `scripts/fetch_external_data.py` |
| `DATASET/clean/` | **ข้อมูลหลัง clean รอบแรก** (สร้างจาก `scripts/clean_data.py` รายละเอียดใน DICTIONARY หัวข้อ 8) |
| `DATASET/_original_raw/` | ไฟล์ต้นฉบับที่ปีปนกัน ก่อนแยก |

ข้อควรรู้: ชื่อไฟล์ใช้ปี พ.ศ. แต่คอลัมน์ปีในไฟล์รายวัน (`dof_*_daily_*`) และใน `external/` เป็น ค.ศ. ไฟล์นอก `clean/` ยังเป็น raw

---