"""
scraper.py - Scrapes transaction history from 28hse.com and caches results to CSV.
"""

import os
import sys
import time
import re
from urllib.parse import urlsplit, urlunsplit, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Ensure stdout handles UTF-8 (Chinese characters) on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_BASE_URL = "https://www.28hse.com/estate/detail/%E5%AF%A7%E5%B3%B0%E8%8B%91-4688/transaction"
MAX_PAGES = 25

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-HK,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.28hse.com/",
}

CHINESE_FLOOR_MAP = {
    "低": 3, "中": 15, "高": 30,
    "低層": 3, "中層": 15, "高層": 30,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
    "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
    "二十六": 26, "二十七": 27, "二十八": 28, "二十九": 29, "三十": 30,
    "三十一": 31, "三十二": 32, "三十三": 33, "三十四": 34, "三十五": 35,
    "三十六": 36, "三十七": 37, "三十八": 38, "三十九": 39, "四十": 40,
}


def normalize_28hse_url(url: str) -> str:
    """Normalize an estate or transaction URL into the transaction-history base URL."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("請輸入 28hse 樓盤連結。")

    if not re.match(r"^https?://", raw, flags=re.IGNORECASE):
        raw = f"https://{raw}"

    parts = urlsplit(raw)
    host = parts.netloc.lower()
    if host.startswith("www."):
        bare_host = host[4:]
    else:
        bare_host = host

    if bare_host != "28hse.com":
        raise ValueError("只支援 28hse.com 連結。")

    path = re.sub(r"/+", "/", parts.path.rstrip("/"))
    if not path:
        raise ValueError("請輸入 28hse 的屋苑或成交頁面連結。")

    path = re.sub(r"/transaction/page-\d+$", "/transaction", path, flags=re.IGNORECASE)

    if "/transaction" in path.lower():
        normalized_path = re.sub(r"/transaction.*$", "/transaction", path, flags=re.IGNORECASE)
    elif path.lower().startswith("/estate/detail/"):
        normalized_path = f"{path}/transaction"
    else:
        raise ValueError("目前只支援 28hse 屋苑詳情頁或成交頁連結。")

    return urlunsplit(("https", "www.28hse.com", normalized_path, "", ""))


def build_page_url(base_url: str, page_num: int) -> str:
    """Build a paginated transaction URL for 28hse."""
    return base_url if page_num == 1 else f"{base_url.rstrip('/')}/page-{page_num}"


def extract_estate_name(url: str) -> str:
    """Best-effort estate name extracted from the 28hse URL."""
    try:
        normalized = normalize_28hse_url(url)
    except ValueError:
        return "目標屋苑"

    match = re.search(r"/estate/detail/([^/]+)/transaction$", normalized)
    if not match:
        return "目標屋苑"

    slug = unquote(match.group(1))
    name = re.sub(r"-\d+$", "", slug).strip()
    return name or "目標屋苑"


def parse_address_cell(text: str) -> dict:
    """
    Parse the address cell text, e.g.:
      '荔欣苑 | 荔影閣 (A 座) 12樓 9室 2026-03-16 註冊處成交'
      '荔欣苑 | 荔欣苑 荔林閣 (C座) 低層 15室 2026-01-19 市場成交 1房'
    Returns dict with: block, floor, flat, date, source
    """
    rec = {}

    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    rec["date"] = m.group(1) if m else None

    # Try multiple block patterns: (A 座), (A座), A座, 荔影閣 (A 座), etc.
    m = re.search(r"\(([A-Z])\s*座\)", text)  # (A 座) or (A座)
    if not m:
        m = re.search(r"([A-Z])\s*座", text)  # A座 or A 座 without parentheses
    if not m:
        m = re.search(r"\(([A-Z])\)", text)  # (A) without 座
    rec["block"] = m.group(1) if m else None

    m_num = re.search(r"(\d+)樓", text)
    if m_num:
        rec["floor"] = int(m_num.group(1))
    else:
        rec["floor"] = None
        for key, val in sorted(CHINESE_FLOOR_MAP.items(), key=lambda x: -len(x[0])):
            if key + "層" in text or key + "樓" in text:
                rec["floor"] = val
                break
            if key in text and len(key) >= 2 and "層" in key:
                rec["floor"] = val
                break

    m_flat = re.search(r"(\d+|[A-Z])室", text)
    rec["flat"] = m_flat.group(1) if m_flat else None

    if "註冊處成交" in text:
        rec["source"] = "registry"
    elif "市場成交" in text:
        rec["source"] = "market"
    else:
        rec["source"] = "unknown"

    return rec


def parse_size_cell(text: str) -> float | None:
    """Parse '實645呎' -> 645.0"""
    m = re.search(r"([\d,]+)\s*呎", text)
    if m:
        return float(m.group(1).replace(",", ""))
    return None


def parse_price_cell(text: str) -> tuple[float | None, float | None]:
    """
    Parse '$593.8 萬元 @ $9,206' -> (5938000.0, 9206.0)
    Returns (price_hkd, price_per_sqft)
    """
    price = None
    per_sqft = None

    m = re.search(r"\$([\d.]+)\s*萬", text)
    if m:
        price = float(m.group(1)) * 10000

    m2 = re.search(r"@\s*\$([\d,]+)", text)
    if m2:
        per_sqft = float(m2.group(1).replace(",", ""))

    return price, per_sqft


def scrape_page(url: str, session: requests.Session) -> list[dict]:
    """Scrape a single page and return list of transaction dicts."""
    resp = session.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    table = soup.find("table", class_="celled")
    if not table:
        tables = soup.find_all("table")
        table = max(tables, key=lambda t: len(t.find_all("tr"))) if tables else None

    if not table:
        return []

    records = []
    rows = table.find_all("tr")
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 3:
            continue

        addr_text = cells[0].get_text(" ", strip=True)
        size_text = cells[1].get_text(" ", strip=True)
        price_text = cells[2].get_text(" ", strip=True)

        rec = parse_address_cell(addr_text)
        rec["size_sqft"] = parse_size_cell(size_text)
        price, per_sqft = parse_price_cell(price_text)
        rec["price"] = price
        rec["price_per_sqft"] = per_sqft

        if rec.get("date") and rec.get("price"):
            records.append(rec)

    return records


def scrape_all_pages(base_url: str = DEFAULT_BASE_URL, max_pages: int = MAX_PAGES) -> pd.DataFrame:
    """Scrape all pages for the chosen estate and return a combined DataFrame."""
    normalized_url = normalize_28hse_url(base_url)
    session = requests.Session()
    all_records = []

    print(f"Scraping up to {max_pages} pages from {normalized_url} ...")

    for page_num in range(1, max_pages + 1):
        url = build_page_url(normalized_url, page_num)
        print(f"  Page {page_num}/{max_pages}: {url}")

        try:
            records = scrape_page(url, session)
        except Exception as e:
            print(f"  Error on page {page_num}: {e}")
            break

        if not records:
            print(f"  No records found on page {page_num}, stopping.")
            break

        all_records.extend(records)
        print(f"  Got {len(records)} records (running total: {len(all_records)})")

        if page_num < max_pages:
            time.sleep(0.5)

    if not all_records:
        print("WARNING: No records scraped.")
        return pd.DataFrame(
            columns=[
                "date", "block", "floor", "flat", "size_sqft",
                "price", "price_per_sqft", "source",
            ]
        )

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["price"]).sort_values("date", ascending=False).reset_index(drop=True)

    # Remove duplicates based on key transaction identifiers
    # Keep registry source over market source when duplicates exist
    print(f"\nTotal records before deduplication: {len(df)}")

    # Sort by source priority (registry first) before dropping duplicates
    source_priority = {"registry": 0, "market": 1, "unknown": 2}
    df["_source_priority"] = df["source"].map(source_priority).fillna(2)
    df = df.sort_values("_source_priority")

    # Drop duplicates based on date, block, floor, flat (same transaction)
    df = df.drop_duplicates(subset=["date", "block", "floor", "flat"], keep="first")
    df = df.drop(columns=["_source_priority"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    print(f"Total records after deduplication: {len(df)}")
    print(f"Removed {len(all_records) - len(df)} duplicate transactions")

    print(f"\nScrape complete. Final unique records: {len(df)}")
    return df


def load_or_scrape(
    cache_path: str = "data/transactions.csv",
    force: bool = False,
    base_url: str = DEFAULT_BASE_URL,
) -> pd.DataFrame:
    """Load from CSV cache if it exists; otherwise scrape and save."""
    normalize_28hse_url(base_url)

    if not force and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"Loaded {len(df)} records from cache.")
        return df

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

    df = scrape_all_pages(base_url=base_url)

    if not df.empty:
        df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} records to {cache_path}")
    else:
        print("No data to save.")

    return df
