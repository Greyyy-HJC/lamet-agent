import json
import os
import time

import requests

FALLBACK_PDF = False

with open("inspirehep.json", "r", encoding="utf-8") as f:
    records = json.load(f)
    arxiv_ids = [
        {
            "id": record["metadata"]["control_number"],
            "arxiv_id": record["metadata"].get("arxiv_eprints", [{}])[0].get("value"),
        }
        for record in records
    ]

os.makedirs("arxiv", exist_ok=True)
with requests.Session() as session:
    session.headers["User-Agent"] = "lamet-agent (local research corpus)"

    for index, arxiv_id in enumerate(arxiv_ids):
        if arxiv_id["arxiv_id"] is None:
            print(f"[{index}/{len(arxiv_ids)}] {arxiv_id['id']}: skipped (no arXiv ID)")
            continue

        file_stem = arxiv_id["arxiv_id"].replace("/", "_")
        html_path = f"arxiv/{file_stem}.html"
        pdf_path = f"arxiv/{file_stem}.pdf"
        md_path = f"arxiv/{file_stem}.md"

        if os.path.exists(html_path) or os.path.exists(pdf_path):
            print(f"[{index}/{len(arxiv_ids)}] {arxiv_id['id']}: skipped (HTML or PDF file exists)")
            continue

        response = session.get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id['arxiv_id']}", timeout=60)
        time.sleep(3)

        if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
            with open(html_path, "wb") as f:
                f.write(response.content)
            print(f"[{index}/{len(arxiv_ids)}] {arxiv_id['id']}: downloaded (HTML)")
        else:
            if FALLBACK_PDF:
                print(f"[{index}/{len(arxiv_ids)}] {arxiv_id['id']}: HTML unavailable, trying PDF")
                response = session.get(f"https://arxiv.org/pdf/{arxiv_id['arxiv_id']}", timeout=60)
                response.raise_for_status()
                time.sleep(3)

                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                print(f"[{index}/{len(arxiv_ids)}] {arxiv_id['id']}: downloaded (PDF)")
            else:
                print(f"[{index}/{len(arxiv_ids)}] {arxiv_id['id']}: skipped (HTML unavailable)")
                continue
