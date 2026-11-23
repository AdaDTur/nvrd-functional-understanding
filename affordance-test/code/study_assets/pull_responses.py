"""
Download the human-study responses from the Google Sheet (via the Apps Script web app's doGet) and
write them out for analysis.

Needs, in the environment or in affordance-test/.env:
    SHEET_READ_KEY   the READ_KEY string set in the Apps Script doGet (keeps the responses private)
    SHEET_URL        optional; defaults to the study's web-app URL (the same one demo.js posts to)

Writes (in affordance-test/):
    responses_raw.csv    the sheet's "responses" tab exactly as stored (received_at, participant, group, results_json)
    responses.csv        same rows, with the attention-check fields removed (the checks were dropped from the study)
    responses_long.csv   one row per participant x object x answer: each part's answer plus each whole-object
                         answer, with the options shown, the selections (joined by "|"), time, visits and the
                         optional tool-use text

Usage (from nvrd/affordance-test/): python3 code/study_assets/pull_responses.py
"""
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))  # affordance-test
DEFAULT_SHEET_URL = ("https://script.google.com/macros/s/AKfycbywIgg7tMNJVidcxyG25I9tU15H0CYPm_BQvs_DGCQIvBmGtjBsiIClNdRVg9ZNy9W5Rw/exec")
COLUMNS = ["received_at", "participant", "group", "results_json"]


def setting(name: str, default: str = "") -> str:
    if os.environ.get(name):
        return os.environ[name]
    try:
        with open(os.path.join(HERE, ".env")) as f:
            for line in f:
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return default


def fetch_rows() -> list:
    key = setting("SHEET_READ_KEY") or sys.exit("SHEET_READ_KEY not set (export it or add it to affordance-test/.env)")
    url = setting("SHEET_URL", DEFAULT_SHEET_URL) + "?" + urllib.parse.urlencode({"key": key})
    with urllib.request.urlopen(url, timeout=120) as resp:  # Apps Script redirects once; urllib follows it
        body = resp.read().decode("utf-8")
    if body.strip() == "forbidden":
        sys.exit("The web app refused the key: check SHEET_READ_KEY matches READ_KEY in the Apps Script.")
    try:
        rows = json.loads(body)
    except ValueError:
        sys.exit("Unexpected reply (is doGet deployed as a new version of the web app?):\n" + body[:500])
    header, data = rows[0], rows[1:]
    if [str(h) for h in header[:4]] != COLUMNS:
        sys.exit(f"Unexpected columns in the responses tab: {header}")
    return [dict(zip(COLUMNS, (str(c) for c in r[:4]))) for r in data if any(str(c).strip() for c in r)]


def write_csv(path: str, fieldnames: list, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    raw = fetch_rows()
    write_csv(os.path.join(HERE, "responses_raw.csv"), COLUMNS, raw)

    clean, long_rows, bad = [], [], []
    for r in raw:
        try:
            d = json.loads(r["results_json"])
        except ValueError:
            bad.append(r["participant"])
            continue
        d.pop("attention_checks", None)
        d.pop("attention_checks_passed", None)
        clean.append({**r, "results_json": json.dumps(d, separators=(",", ":"))})
        base = {"participant": r["participant"], "group": d.get("group_index", r["group"]),
                "received_at": r["received_at"], "descriptions_shown": d.get("descriptions_shown"),
                "total_minutes": round((d.get("total_duration_ms") or 0) / 60000, 1)}
        for o in d.get("objects", []):
            obj = {**base, "object": o["object"], "size": o.get("size_chart_scale")}
            for reg in o.get("regions", []):
                long_rows.append({**obj, "question": "part", "part": reg.get("part"), "region_id": reg.get("region_id"),
                                  "selected": "|".join(reg.get("functionalities_selected") or []),
                                  "options_shown": "|".join(reg.get("options_shown") or []),
                                  "response_time_ms": reg.get("response_time_ms"), "visits": reg.get("visits"),
                                  "tool_interaction": reg.get("tool_interaction") or ""})
            long_rows.append({**obj, "question": "overall", "part": "", "region_id": "",
                              "selected": "|".join(o.get("overall_functionalities_selected") or []),
                              "options_shown": "|".join(o.get("overall_options_shown") or []),
                              "response_time_ms": o.get("overall_response_time_ms"), "visits": o.get("overall_visits"),
                              "tool_interaction": ""})

    write_csv(os.path.join(HERE, "responses.csv"), COLUMNS, clean)
    long_cols = ["participant", "group", "received_at", "descriptions_shown", "total_minutes", "object", "size",
                 "question", "part", "region_id", "selected", "options_shown", "response_time_ms", "visits",
                 "tool_interaction"]
    write_csv(os.path.join(HERE, "responses_long.csv"), long_cols, long_rows)

    per_group = {}
    for r in clean:
        per_group.setdefault(r["group"], []).append(r["participant"])
    print(f"{len(clean)} responses -> responses.csv, responses_raw.csv, responses_long.csv ({len(long_rows)} answers)")
    for g in sorted(per_group, key=lambda x: int(x) if str(x).isdigit() else 99):
        print(f"  group {g}: {len(per_group[g])}  ({', '.join(per_group[g])})")
    if bad:
        print(f"  skipped (unreadable results_json): {', '.join(bad)}")


if __name__ == "__main__":
    main()
