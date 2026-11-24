"""从 10 份官方精算费率表 PDF 解析结构化费率项并生成 data/parsed/rate_tables.json。

覆盖真实产品库中具备官方费率表 PDF 的全部 11 款产品（ID 2–12）：
- ID 2: 复星联合优选K重疾险(达尔文12号)费率表.pdf（终身计划1，20年/30年交，每千元保额费率）
- ID 3: 复星联合医联有盟重疾险费率表.pdf（计划1有医保/计划5无医保，每万元保额费率，10万起）
- ID 4/5: 复星联合优选二号长期住院医疗险(星相守2号)费率表.pdf（ID 4 有等待期个人版 / ID 5 无等待期转保版）
- ID 6: 众安尊享e生百万医疗险2026版费率表.pdf（年龄分段有/无社保年交费率）
- ID 7: 中英人寿福满佳C款(悦享版)终身寿险分红型费率表.pdf（每千元保费对应基本保额，5年交）
- ID 8: 太平洋福有余2025终身寿险分红型费率表.pdf（每千元保费对应基本保额，5年交）
- ID 9: 新华人寿E增福优享版终身寿险互联网费率表.pdf（每千元保费对应基本保额，5年交）
- ID 10: 太平洋小蜜蜂6号综合意外险费率表.pdf（固定计划保费）
- ID 11: 太平洋小蜜蜂6号玫瑰版综合意外险费率表.pdf（固定计划保费）
- ID 12: 人保小学童2号Pro学平险费率表.pdf（固定计划保费）
注：ID 1（达尔文宝贝计划12号）未提供官方费率表 PDF，不在本文件造假。

用法：在 agent-service 目录执行
    uv run python scripts/parse_rate_tables.py
"""

from pathlib import Path
import json
import re
from typing import Any

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RATE_DIR = PROJECT_ROOT / "data" / "raw" / "rate_tables"
OUTPUT_PATH = PROJECT_ROOT / "data" / "parsed" / "rate_tables.json"


def _row(
    *,
    plan_code: str,
    rate_type: str,
    age_min: int | None,
    age_max: int | None,
    gender: str = "all",
    occupation_class: str = "all",
    social_security: str = "all",
    payment_period: str | None = None,
    coverage_period: str | None = None,
    coverage_amount: float | None = None,
    deductible: float | None = None,
    rate_value: float | None = None,
    premium: float | None = None,
    source_page: int | None = None,
    source_table: str = "",
) -> dict[str, Any]:
    return {
        "plan_code": plan_code,
        "rate_type": rate_type,
        "age_min": age_min,
        "age_max": age_max,
        "gender": gender,
        "occupation_class": occupation_class,
        "social_security": social_security,
        "payment_period": payment_period,
        "coverage_period": coverage_period,
        "coverage_amount": coverage_amount,
        "deductible": deductible,
        "rate_value": rate_value,
        "premium": premium,
        "source_page": source_page,
        "source_table": source_table,
    }


def parse_darwin12_id2() -> list[dict[str, Any]]:
    """ID 2: 达尔文12号（复星联合优选K重疾险，终身 计划1 必选责任，20年交 & 30年交，0-35岁）。"""
    pdf_path = RAW_RATE_DIR / "复星联合优选K重疾险(达尔文12号)费率表.pdf"
    reader = PdfReader(pdf_path)

    pay_map = {"20年交": "20y", "30年交": "30y"}
    gender_map = {"男": "male", "女": "female"}
    pattern = re.compile(
        r"^终身\s+1\s+([男女])\s+(20年交|30年交)\s+(\d+)\s+([\d.]+)\s+"
    )
    collected: dict[tuple[str, str, int], tuple[float, int]] = {}

    # 优先扫描目标页窗（男20y/30y、女20y/30y 约在 p225-236, p293-305, p632-643, p705-720），缺失再兜底全扫
    expected_keys = 2 * 2 * 36
    priority_pages: list[int] = []
    seen_pages: set[int] = set()
    for rng in (
        range(228, 234),
        range(296, 302),
        range(635, 641),
        range(700, 715),
    ):
        for idx in rng:
            if 0 <= idx < len(reader.pages) and idx not in seen_pages:
                priority_pages.append(idx)
                seen_pages.add(idx)

    fallback_pages = [
        idx
        for idx in range(min(215, len(reader.pages) - 1), min(735, len(reader.pages)))
        if idx not in seen_pages
    ]

    for page_idx in [*priority_pages, *fallback_pages]:
        text = reader.pages[page_idx].extract_text() or ""
        if "终身 1 " not in text:
            continue
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            g_cn, pay_cn, age_s, rate_s = m.groups()
            age = int(age_s)
            if age > 35:
                continue
            gender = gender_map[g_cn]
            pay_code = pay_map[pay_cn]
            key = (pay_code, gender, age)
            if key not in collected:
                collected[key] = (float(rate_s), page_idx + 1)
        if len(collected) >= expected_keys:
            break

    items: list[dict[str, Any]] = []
    # 排序保证 20y 在 30y 之前（默认 standard 优先匹配 20y，与产品主交费期一致）
    for (pay_code, gender, age), (rate_val, pno) in sorted(collected.items()):
        items.append(
            _row(
                plan_code="standard",
                rate_type="per_1000_sum_assured",
                age_min=age,
                age_max=age,
                gender=gender,
                occupation_class="1-4",
                social_security="all",
                payment_period=pay_code,
                coverage_period="lifetime",
                coverage_amount=100000.0,
                rate_value=rate_val,
                source_page=pno,
                source_table=f"终身计划1_{pay_code}_每千元基本保额",
            )
        )
    return items


def parse_yilianyoumeng_id3() -> list[dict[str, Any]]:
    """ID 3: 复星联合医联有盟重疾险（100000基础保额表，计划1有医保/计划5无医保，18-59岁）。"""
    pdf_path = RAW_RATE_DIR / "复星联合医联有盟重疾险费率表.pdf"
    reader = PdfReader(pdf_path)

    pattern = re.compile(
        r"^100000\s+([15])\s+([男女])\s+(20年交|15年交|10年交)\s+(\d+)\s+([\d.]+)$"
    )
    pay_map = {"20年交": "20y", "15年交": "15y", "10年交": "10y"}
    pay_priority = {"20y": 0, "15y": 1, "10y": 2}
    gender_map = {"男": "male", "女": "female"}
    ss_map = {"1": "yes", "5": "no"}

    # 100000 档分布在 pages 2..60
    raw_hits: dict[tuple[int, str, str, str], tuple[float, int]] = {}
    for page_idx in range(1, min(65, len(reader.pages))):
        text = reader.pages[page_idx].extract_text() or ""
        if "100000 " not in text:
            continue
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if not m:
                continue
            plan_num, g_cn, pay_cn, age_s, prem_s = m.groups()
            age = int(age_s)
            if not (18 <= age <= 59):
                continue
            key = (age, gender_map[g_cn], ss_map[plan_num], pay_map[pay_cn])
            raw_hits[key] = (float(prem_s), page_idx + 1)

    items: list[dict[str, Any]] = []
    ordered = sorted(
        raw_hits.items(),
        key=lambda kv: (pay_priority[kv[0][3]], kv[0][0], kv[0][1], kv[0][2]),
    )
    for (age, gender, ss, pay_code), (prem_100k, pno) in ordered:
        items.append(
            _row(
                plan_code="standard",
                rate_type="per_10000_sum_assured",
                age_min=age,
                age_max=age,
                gender=gender,
                occupation_class="1-4",
                social_security=ss,
                payment_period=pay_code,
                coverage_period="lifetime",
                coverage_amount=100000.0,
                rate_value=round(prem_100k / 10.0, 4),
                source_page=pno,
                source_table=f"计划{'一(有医保)' if ss == 'yes' else '五(无医保)'}_{pay_code}_10万基础表",
            )
        )
    return items


def parse_xingxiangshou2_id4_id5() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """ID 4 (个人版-有等待期) & ID 5 (转保版-无等待期): 星相守2号长期医疗险第 2-3 页基准表。"""
    pdf_path = RAW_RATE_DIR / "复星联合优选二号长期住院医疗险(星相守2号)费率表.pdf"
    reader = PdfReader(pdf_path)

    items_id4: list[dict[str, Any]] = []
    items_id5: list[dict[str, Any]] = []

    for page_idx in (1, 2):
        text = reader.pages[page_idx].extract_text() or ""
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            age = int(parts[0])
            if len(parts) != 9:
                continue
            vals = parts[1:]

            # ID 4 个人版（有等待期）：产品年龄范围 1-55 岁，保留 0-70 岁基准费率表便于核对
            if 0 <= age <= 55:
                p1_w_ss, p1_w_noss, p2_w_ss, p2_w_noss = vals[0], vals[1], vals[2], vals[3]
                for ss, val_str in (("yes", p1_w_ss), ("no", p1_w_noss)):
                    if val_str != "-":
                        items_id4.append(
                            _row(
                                plan_code="standard",
                                rate_type="age_band_plan_premium",
                                age_min=age,
                                age_max=age,
                                gender="all",
                                occupation_class="1-4",
                                social_security=ss,
                                payment_period="annual",
                                coverage_period="1y",
                                coverage_amount=4000000.0,
                                deductible=10000.0,
                                premium=float(val_str),
                                source_page=page_idx + 1,
                                source_table="必选责任基准表_有等待期计划一",
                            )
                        )
                for ss, val_str in (("yes", p2_w_ss), ("no", p2_w_noss)):
                    if val_str != "-":
                        items_id4.append(
                            _row(
                                plan_code="plan2",
                                rate_type="age_band_plan_premium",
                                age_min=age,
                                age_max=age,
                                gender="all",
                                occupation_class="1-4",
                                social_security=ss,
                                payment_period="annual",
                                coverage_period="1y",
                                coverage_amount=4000000.0,
                                deductible=0.0,
                                premium=float(val_str),
                                source_page=page_idx + 1,
                                source_table="必选责任基准表_有等待期计划二(0免赔)",
                            )
                        )

            # ID 5 转保版（无等待期）：1-70 岁
            if 1 <= age <= 70:
                p1_nw_ss, p1_nw_noss, p2_nw_ss, p2_nw_noss = vals[4], vals[5], vals[6], vals[7]
                for ss, val_str in (("yes", p1_nw_ss), ("no", p1_nw_noss)):
                    if val_str != "-":
                        items_id5.append(
                            _row(
                                plan_code="standard",
                                rate_type="age_band_plan_premium",
                                age_min=age,
                                age_max=age,
                                gender="all",
                                occupation_class="1-4",
                                social_security=ss,
                                payment_period="annual",
                                coverage_period="1y",
                                coverage_amount=4000000.0,
                                deductible=None,
                                premium=float(val_str),
                                source_page=page_idx + 1,
                                source_table="必选责任基准表_无等待期计划一",
                            )
                        )
                for ss, val_str in (("yes", p2_nw_ss), ("no", p2_nw_noss)):
                    if val_str != "-":
                        items_id5.append(
                            _row(
                                plan_code="plan2",
                                rate_type="age_band_plan_premium",
                                age_min=age,
                                age_max=age,
                                gender="all",
                                occupation_class="1-4",
                                social_security=ss,
                                payment_period="annual",
                                coverage_period="1y",
                                coverage_amount=4000000.0,
                                deductible=0.0,
                                premium=float(val_str),
                                source_page=page_idx + 1,
                                source_table="必选责任基准表_无等待期计划二(0免赔)",
                            )
                        )

    # 补充 56-70 岁 ID 4 基准费率表条目（供单年龄查表对照，产品目录硬过滤仍按 min_age/max_age 限制）
    p3_text = reader.pages[2].extract_text() or ""
    for line in p3_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 9 and parts[0].isdigit():
            age = int(parts[0])
            if 56 <= age <= 70:
                for ss, val_str in (("yes", parts[1]), ("no", parts[2])):
                    items_id4.append(
                        _row(
                            plan_code="standard",
                            rate_type="age_band_plan_premium",
                            age_min=age,
                            age_max=age,
                            gender="all",
                            occupation_class="1-4",
                            social_security=ss,
                            payment_period="annual",
                            coverage_period="1y",
                            coverage_amount=4000000.0,
                            deductible=10000.0,
                            premium=float(val_str),
                            source_page=3,
                            source_table="必选责任基准表_有等待期计划一",
                        )
                    )
                for ss, val_str in (("yes", parts[3]), ("no", parts[4])):
                    items_id4.append(
                        _row(
                            plan_code="plan2",
                            rate_type="age_band_plan_premium",
                            age_min=age,
                            age_max=age,
                            gender="all",
                            occupation_class="1-4",
                            social_security=ss,
                            payment_period="annual",
                            coverage_period="1y",
                            coverage_amount=4000000.0,
                            deductible=0.0,
                            premium=float(val_str),
                            source_page=3,
                            source_table="必选责任基准表_有等待期计划二(0免赔)",
                        )
                    )

    items_id4.sort(key=lambda x: (x["plan_code"] != "standard", x["age_min"], x["social_security"]))
    items_id5.sort(key=lambda x: (x["plan_code"] != "standard", x["age_min"], x["social_security"]))
    return items_id4, items_id5


def parse_zunxiangesheng_id6() -> list[dict[str, Any]]:
    """ID 6: 众安尊享e生百万医疗险2026版（第 1 页年龄分段有/无社保年交表）。"""
    pdf_path = RAW_RATE_DIR / "众安尊享e生百万医疗险2026版费率表.pdf"
    reader = PdfReader(pdf_path)
    text = reader.pages[0].extract_text() or ""
    pattern = re.compile(r"^\[(\d+),(\d+)\]\s+([\d.]+)\s+([\d.]+)$")

    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        age_min_s, age_max_s, ss_yes_s, ss_no_s = m.groups()
        age_min, age_max = int(age_min_s), int(age_max_s)
        for ss_flag, prem_s in (("yes", ss_yes_s), ("no", ss_no_s)):
            items.append(
                _row(
                    plan_code="standard",
                    rate_type="age_band_plan_premium",
                    age_min=age_min,
                    age_max=age_max,
                    gender="all",
                    occupation_class="1-6",
                    social_security=ss_flag,
                    payment_period="annual",
                    coverage_period="1y",
                    coverage_amount=1000000.0,
                    deductible=10000.0,
                    premium=float(prem_s),
                    source_page=1,
                    source_table="尊享e生2026必选计划年缴表",
                )
            )
    return items


def _extract_life_pair(product_id: int, parts: list[str]) -> tuple[str, str, str] | None:
    """根据各终身寿险费率表高龄退阶列数规则，提取 (payment_period, male_val, female_val)。"""
    n = len(parts)
    if product_id == 7:
        if n in (11, 7):
            return ("5y", parts[5], parts[6])
        if n == 5:
            return ("3y", parts[3], parts[4])
        if n == 3:
            return ("single", parts[1], parts[2])
        return None
    if product_id == 8:
        if n == 11:
            return ("5y", parts[3], parts[8])
        if n == 9:
            return ("5y", parts[3], parts[7])
        if n == 7:
            return ("5y", parts[3], parts[6])
        if n == 5:
            return ("3y", parts[2], parts[4])
        if n == 3:
            return ("single", parts[1], parts[2])
        return None
    if product_id == 9:
        if n == 7:
            return ("5y", parts[3], parts[6])
        if n == 5:
            return ("3y", parts[2], parts[4])
        if n == 3:
            return ("single", parts[1], parts[2])
        return None
    return None


def parse_life_rate_table(
    *,
    product_id: int,
    filename: str,
    table_title: str,
    min_premium: float,
    max_age_limit: int,
) -> list[dict[str, Any]]:
    """解析终身寿险每 1,000 元年交保费对应基本保额表（优先 5 年交，高龄自动衔接 3 年交/趸交）。"""
    pdf_path = RAW_RATE_DIR / filename
    reader = PdfReader(pdf_path)
    items: list[dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        for line in text.splitlines():
            clean = line.strip()
            if not clean:
                continue
            # 兼容如 "0（须出生满 30 日） 964 2840 ..." 的首行说明
            clean = re.sub(r"^(\d+)（[^）]+）", r"\1", clean)
            parts = clean.split()
            if not parts or not parts[0].isdigit():
                continue
            age = int(parts[0])
            if age > max_age_limit:
                continue
            extracted = _extract_life_pair(product_id, parts)
            if extracted is None:
                continue
            pay_period, male_val, female_val = extracted
            for gender, val_s in (("male", male_val), ("female", female_val)):
                if not re.fullmatch(r"\d+(?:\.\d+)?", val_s):
                    continue
                items.append(
                    _row(
                        plan_code="standard",
                        rate_type="per_1000_premium_sum_assured",
                        age_min=age,
                        age_max=age,
                        gender=gender,
                        occupation_class="1-6",
                        social_security="all",
                        payment_period=pay_period,
                        coverage_period="lifetime",
                        rate_value=float(val_s),
                        premium=min_premium,
                        source_page=page_idx + 1,
                        source_table=f"{table_title}_{pay_period}",
                    )
                )
    items.sort(key=lambda x: (x["age_min"], x["gender"]))
    return items


def parse_fixed_accident_tables() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """ID 10 小蜜蜂6号 / ID 11 玫瑰版 / ID 12 小学童2号Pro 固定计划费率表。"""
    # ID 10: 18-50 周岁 经典136 / 典藏166 / 优享236 / 尊享298
    p10 = (PdfReader(RAW_RATE_DIR / "太平洋小蜜蜂6号综合意外险费率表.pdf").pages[0].extract_text() or "")
    m10 = re.search(r"18-50\s*周岁（含）\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", p10)
    if not m10:
        raise ValueError("未能解析太平洋小蜜蜂6号综合意外险费率表")
    p10_vals = [float(x) for x in m10.groups()]
    items_id10 = [
        _row(
            plan_code=code,
            rate_type="fixed_plan_premium",
            age_min=18,
            age_max=50,
            gender="all",
            occupation_class="1-3",
            social_security="all",
            payment_period="annual",
            coverage_period="1y",
            coverage_amount=cov,
            premium=prem,
            source_page=1,
            source_table=f"小蜜蜂6号_{label}",
        )
        for code, label, cov, prem in zip(
            ("standard", "deluxe", "premium", "supreme"),
            ("经典版", "典藏版", "优享版", "尊享版"),
            (300000.0, 500000.0, 1000000.0, 1500000.0),
            p10_vals,
            strict=True,
        )
    ]

    # ID 11: 18-50 周岁 典藏169 / 尊享319 / 至尊439
    p11 = (PdfReader(RAW_RATE_DIR / "太平洋小蜜蜂6号玫瑰版综合意外险费率表.pdf").pages[0].extract_text() or "")
    m11 = re.search(r"18-50\s*周岁（含）\s+(\d+)\s+(\d+)\s+(\d+)", p11)
    if not m11:
        raise ValueError("未能解析太平洋小蜜蜂6号玫瑰版综合意外险费率表")
    p11_vals = [float(x) for x in m11.groups()]
    items_id11 = [
        _row(
            plan_code=code,
            rate_type="fixed_plan_premium",
            age_min=18,
            age_max=50,
            gender="all",
            occupation_class="1-3",
            social_security="all",
            payment_period="annual",
            coverage_period="1y",
            coverage_amount=cov,
            premium=prem,
            source_page=1,
            source_table=f"小蜜蜂6号玫瑰版_{label}",
        )
        for code, label, cov, prem in zip(
            ("standard", "premium", "supreme"),
            ("典藏版", "尊享版", "至尊版"),
            (500000.0, 1000000.0, 1500000.0),
            p11_vals,
            strict=True,
        )
    ]

    # ID 12: 3-24 周岁 基础版180 / 经典版240 / 尊贵版335
    p12 = (PdfReader(RAW_RATE_DIR / "人保小学童2号Pro学平险费率表.pdf").pages[0].extract_text() or "")
    m12 = re.search(r"3-24\s*周岁\s+1\s*年\s+(\d+)\s*元\s+(\d+)\s*元\s+(\d+)\s*元", p12)
    if not m12:
        raise ValueError("未能解析人保小学童2号Pro学平险费率表")
    p12_basic, p12_classic, p12_premium = [float(x) for x in m12.groups()]
    items_id12 = [
        _row(
            plan_code=code,
            rate_type="fixed_plan_premium",
            age_min=3,
            age_max=24,
            gender="all",
            occupation_class="1-4",
            social_security="all",
            payment_period="annual",
            coverage_period="1y",
            coverage_amount=cov,
            premium=prem,
            source_page=1,
            source_table=f"小学童2号Pro_{label}",
        )
        for code, label, cov, prem in (
            ("standard", "经典版", 200000.0, p12_classic),
            ("basic", "基础版", 100000.0, p12_basic),
            ("premium", "尊贵版", 300000.0, p12_premium),
        )
    ]

    return items_id10, items_id11, items_id12


def main() -> None:
    id4_items, id5_items = parse_xingxiangshou2_id4_id5()
    id10_items, id11_items, id12_items = parse_fixed_accident_tables()

    tables: dict[str, list[dict[str, Any]]] = {
        "2": parse_darwin12_id2(),
        "3": parse_yilianyoumeng_id3(),
        "4": id4_items,
        "5": id5_items,
        "6": parse_zunxiangesheng_id6(),
        "7": parse_life_rate_table(
            product_id=7,
            filename="中英人寿福满佳C款(悦享版)终身寿险分红型费率表.pdf",
            table_title="福满佳C款(悦享版)_每千元保费对应基本保额",
            min_premium=10000.0,
            max_age_limit=57,
        ),
        "8": parse_life_rate_table(
            product_id=8,
            filename="太平洋福有余2025终身寿险分红型费率表.pdf",
            table_title="太保福有余2025_每千元保费对应基本保额",
            min_premium=5000.0,
            max_age_limit=68,
        ),
        "9": parse_life_rate_table(
            product_id=9,
            filename="新华人寿E增福优享版终身寿险互联网费率表.pdf",
            table_title="新华E增福优享版_每千元保费对应基本保额",
            min_premium=10000.0,
            max_age_limit=70,
        ),
        "10": id10_items,
        "11": id11_items,
        "12": id12_items,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(tables, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"已解析生成 {OUTPUT_PATH}（共 {len(tables)}/12 款真实产品官方费率表）：")
    for pid, rows in tables.items():
        if not rows:
            raise RuntimeError(f"产品 ID={pid} 解析结果为空！")
        print(f"  - ID={pid:>2}: {len(rows):>3} 条记录 | 示例: {rows[0]}")


if __name__ == "__main__":
    main()
