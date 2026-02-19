from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(rows: Sequence[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def zip_single_file(file_path: str | Path, archive_path: str | Path, arcname: str = "prediction.json") -> None:
    from zipfile import ZIP_DEFLATED, ZipFile

    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
        zf.write(file_path, arcname=arcname)


def to_qrel_map(qrels: Iterable[dict]) -> dict[str, set[str]]:
    rel: dict[str, set[str]] = {}
    for row in qrels:
        if int(row.get("qrel", 0)) <= 0:
            continue
        qid = str(row["qid"])
        rel.setdefault(qid, set()).add(str(row["docid"]))
    return rel
