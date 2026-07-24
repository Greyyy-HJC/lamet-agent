from __future__ import annotations

import http.client
import random
import socket
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.error import HTTPError, URLError


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}
OPENSEARCH_NS = {"opensearch": "http://a9.com/-/spec/opensearch/1.1/"}


class ArxivRateLimitError(RuntimeError):
    """Raised when the arXiv API explicitly reports rate limiting."""


def compute_backoff_delay(error: Exception, attempt: int, base_backoff_seconds: float) -> float:
    multiplier = 2 ** (attempt - 1)
    jitter = random.uniform(0.0, 0.75)
    if isinstance(error, HTTPError) and error.code == 429:
        return max(60.0, base_backoff_seconds * 10 * multiplier) + jitter
    if isinstance(error, ArxivRateLimitError):
        return max(60.0, base_backoff_seconds * 10 * multiplier) + jitter
    return base_backoff_seconds * multiplier + jitter


def iso_date_to_query_floor(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d0000")


def iso_date_to_query_ceiling(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d2359")


def build_query(raw_query: str, start_date: str, end_date: str) -> str:
    date_clause = "submittedDate:[{start} TO {end}]".format(
        start=iso_date_to_query_floor(start_date),
        end=iso_date_to_query_ceiling(end_date),
    )
    return f"({raw_query}) AND {date_clause}"


def extract_id_components(entry_id: str) -> Dict[str, Any]:
    short = entry_id.rstrip("/").split("/")[-1]
    if "v" in short and short.rsplit("v", 1)[-1].isdigit():
        base, version_text = short.rsplit("v", 1)
        return {"arxiv_id": base, "latest_version": int(version_text)}
    return {"arxiv_id": short, "latest_version": 1}


def parse_entry(entry: ET.Element) -> Dict[str, Any]:
    raw_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
    id_bits = extract_id_components(raw_id)
    authors = [
        author.findtext("atom:name", default="", namespaces=ATOM_NS).strip()
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    categories = [item.attrib.get("term", "").strip() for item in entry.findall("atom:category", ATOM_NS)]
    links = entry.findall("atom:link", ATOM_NS)
    pdf_url = ""
    for link in links:
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            pdf_url = link.attrib.get("href", "").strip()
            break

    primary_category_node = entry.find("arxiv:primary_category", ARXIV_NS)
    primary_category = ""
    if primary_category_node is not None:
        primary_category = primary_category_node.attrib.get("term", "").strip()

    record: Dict[str, Any] = {
        "arxiv_id": id_bits["arxiv_id"],
        "latest_version": id_bits["latest_version"],
        "title": " ".join(entry.findtext("atom:title", default="", namespaces=ATOM_NS).split()),
        "summary": " ".join(entry.findtext("atom:summary", default="", namespaces=ATOM_NS).split()),
        "authors": [name for name in authors if name],
        "primary_category": primary_category,
        "categories": [term for term in categories if term],
        "published": entry.findtext("atom:published", default="", namespaces=ATOM_NS).strip(),
        "updated": entry.findtext("atom:updated", default="", namespaces=ATOM_NS).strip(),
        "comment": entry.findtext("arxiv:comment", default="", namespaces=ARXIV_NS).strip(),
        "journal_ref": entry.findtext("arxiv:journal_ref", default="", namespaces=ARXIV_NS).strip(),
        "doi": entry.findtext("arxiv:doi", default="", namespaces=ARXIV_NS).strip(),
        "abs_url": raw_id,
        "pdf_url": pdf_url,
        "query_hits": [],
    }
    return record


def fetch_page(
    api_url: str,
    query: str,
    start: int,
    max_results: int,
    timeout: int = 60,
    max_attempts: int = 8,
    base_backoff_seconds: float = 3.0,
) -> Dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
    )
    request = urllib.request.Request(
        f"{api_url}?{params}",
        headers={"User-Agent": "lamet-kb/1.0 (metadata harvest)"},
    )
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if b"Rate exceeded." in payload:
                raise ArxivRateLimitError("arXiv API rate exceeded")
            break
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                raise
        except URLError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
        except socket.timeout as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
        except http.client.RemoteDisconnected as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
        except ConnectionResetError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
        except ArxivRateLimitError as exc:
            last_error = exc
            if attempt == max_attempts:
                raise

        delay = compute_backoff_delay(last_error, attempt, base_backoff_seconds)
        time.sleep(delay)
    else:
        if last_error is not None:
            raise last_error
        raise RuntimeError("arXiv fetch failed without a captured error")

    root = ET.fromstring(payload)
    total_text = root.findtext("opensearch:totalResults", default="0", namespaces=OPENSEARCH_NS)
    total_results = int(total_text)
    entries = [parse_entry(entry) for entry in root.findall("atom:entry", ATOM_NS)]
    return {"total_results": total_results, "entries": entries}


def iter_query_results(
    api_url: str,
    raw_query: str,
    start_date: str,
    end_date: str,
    page_size: int,
    sleep_seconds: float,
    max_results_per_query: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    compiled_query = build_query(raw_query, start_date, end_date)
    start = 0
    yielded = 0

    while True:
        remaining = None if max_results_per_query is None else max_results_per_query - yielded
        if remaining is not None and remaining <= 0:
            break

        page_limit = page_size if remaining is None else min(page_size, remaining)
        page = fetch_page(api_url=api_url, query=compiled_query, start=start, max_results=page_limit)
        entries = page["entries"]
        if not entries:
            break

        for entry in entries:
            yield entry
            yielded += 1

        start += len(entries)
        if start >= page["total_results"]:
            break
        time.sleep(sleep_seconds)


def iter_date_windows(start_date: str, end_date: str, window_days: int) -> Iterator[Tuple[str, str]]:
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    final = datetime.strptime(end_date, "%Y-%m-%d").date()
    step = timedelta(days=window_days)

    while current <= final:
        window_end = min(current + step - timedelta(days=1), final)
        yield current.isoformat(), window_end.isoformat()
        current = window_end + timedelta(days=1)


def today_iso() -> str:
    return date.today().isoformat()
