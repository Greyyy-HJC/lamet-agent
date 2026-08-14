import json

import requests

fields = ",".join(
    [
        "titles",
        "abstracts",
        "arxiv_eprints",
    ]
)
abstracts_values = " or ".join(
    f'abstracts.value: "{v}"'
    for v in [
        "LaMET",
        "large-momentum effective theory",
        "large momentum effective theory",
        "quasi-PDF",
        "quasi-parton",
        "quasi-distribution",
        "quasi-DA",
        "quasi-GPD",
        "quasi-TMD",
    ]
)

with requests.Session() as session:
    records = []
    url = "https://inspirehep.net/api/literature"
    params = {
        "q": f"primarch:hep-lat and ({abstracts_values})",
        "size": 100,
        "sort": "mostrecent",
        "fields": fields,
    }
    while url:
        r = session.get(url, params)
        r.raise_for_status()

        data = r.json()

        records.extend(data["hits"]["hits"])
        url = data.get("links", {}).get("next")
        params = None

with open("inspirehep.json", "w") as f:
    json.dump(records, f, indent=4)
