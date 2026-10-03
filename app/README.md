# Thai Seafood Export Outlook — Streamlit app

แอปแสดงผลลัพธ์ของโครงงาน ML (`../main.ipynb`) ใช้ **โมเดลจริง** (LightGBM) ทำนายสดได้ใน 6 หน้า:
ภาพรวม · กลุ่มตลาด · พยากรณ์ (ย้อนหลัง + ล่วงหน้า พร้อมช่วง 80%) · Early Warning · สถานการณ์จำลอง (what-if) · วิธีอ่านและข้อจำกัด

## รัน

```bash
# จากโฟลเดอร์โปรเจกต์ (IMPORT-EXPORT)
source .venv/bin/activate            # หรือสร้าง venv ใหม่
pip install -r app/requirements.txt
streamlit run app/app.py
```
เปิดเบราว์เซอร์ที่ <http://localhost:8501> (macOS: ถ้า `lightgbm` ขึ้น error เรื่อง `libomp` ให้รัน `brew install libomp`)

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | หน้าจอทั้ง 6 หน้า |
| `core.py` | สร้างตัวแปรจากประวัติ 1 อนุกรม + ทำนายด้วยโมเดล (สูตรเหมือน `main.ipynb` ทุกตัว) |
| `data/` | ข้อมูลที่แอปใช้ (ประวัติรายเดือน, ผลทดสอบย้อนหลัง, พยากรณ์ล่วงหน้า, ผล Early Warning, กลุ่มตลาด) |
| `models/` | โมเดล LightGBM ที่เทรนแล้ว (point + quantile 10/90% ต่อระยะ 1/3/6 เดือน + Early Warning) และ `model_meta.json` |

## สร้างไฟล์ `data/` และ `models/` ใหม่ (เช่น หลังอัปเดตข้อมูล)

```bash
python scripts/build_app_assets.py   # เทรนโมเดลสุดท้าย พยากรณ์ล่วงหน้า และตรวจว่าตรงกับ notebook
python scripts/test_app_features.py  # ทดสอบว่า core.py ให้ผลตรงกับ notebook ทุกอนุกรม
```
(ต้องมีผลจาก `main.ipynb` ใน `DATASET/prepared/` ก่อน)

## หมายเหตุ

- ข้อมูลถึง เม.ย. 2026 (UN Comtrade, USD) — พยากรณ์ข้างหน้ายังไม่รู้ผลจริง
- หน้า Early Warning ใช้ LightGBM (PR-AUC 0.832 ในช่วงทดสอบ) ส่วนรายการใน `DATASET/prepared/early_warning_latest.csv` ของ notebook ใช้ Random Forest (0.827) ผลต่างกันเล็กน้อย
- โมเดลไม่รู้เหตุการณ์ภายนอก (ภาษี การห้ามนำเข้า) สถานการณ์จำลองจึงเป็นการทดลองว่าโมเดลตอบสนองอย่างไร ไม่ใช่การพยากรณ์เหตุการณ์จริง
- แนะนำเทรนโมเดลใหม่ทุก 6–12 เดือน และปรับเทียบช่วง 80% ใหม่ปีละครั้ง
