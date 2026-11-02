"""
Web app tests: the page and assets served by Flask, plus the endpoints
added for it (/pivot, /data, /data.csv, /report).

Data-backed tests need the raw dataset and skip cleanly without it.
"""

import io

import pandas as pd
import pytest

from analysis.data_loader import DEFAULT_RAW_PATH

pytest.importorskip("flask")

needs_data = pytest.mark.skipif(not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present")


@pytest.fixture(scope="module")
def client():
    from api import create_app
    return create_app().test_client()


# ---------------------------------------------------------------------
# Page and assets
# ---------------------------------------------------------------------

def test_index_serves_the_web_app(client):
    r = client.get("/")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert "<title>Finlatics Ad Analytics</title>" in html
    for src in ("/ui/app.css", "/ui/app.js", "/ui/markdown.js", "/ui/vendor/plotly.min.js"):
        assert src in html


@pytest.mark.parametrize("path", ["/ui/app.css", "/ui/app.js", "/ui/markdown.js"])
def test_assets_are_served(client, path):
    assert client.get(path).status_code == 200


def test_plotly_is_served_locally(client):
    pytest.importorskip("plotly")
    r = client.get("/ui/vendor/plotly.min.js")
    assert r.status_code == 200
    assert "javascript" in r.mimetype
    assert b"plotly" in r.data[:500].lower()


def test_unknown_asset_is_404(client):
    assert client.get("/ui/nope.js").status_code == 404


# ---------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------

def test_reports_are_served_as_markdown(client):
    r = client.get("/report/eda")
    assert r.status_code == 200
    assert r.mimetype == "text/markdown"
    text = r.get_data(as_text=True)
    for q in (1, 10, 20):
        assert f"## Q{q} ·" in text

    assert client.get("/report/model").status_code == 200
    assert client.get("/report/secret").status_code == 404


def test_report_figures_are_served_without_path_traversal(client):
    r = client.get("/report/figures/01_daily_trends.png")
    assert r.status_code == 200 and r.mimetype == "image/png"
    assert client.get("/report/figures/../eda_findings.json").status_code == 404
    assert client.get("/report/figures/%2e%2e/eda_findings.json").status_code == 404


# ---------------------------------------------------------------------
# Pivot
# ---------------------------------------------------------------------

@needs_data
def test_pivot_matches_overall_totals(client):
    body = client.get("/pivot?rows=banner&cols=placement").get_json()
    assert body["rows"] == "banner" and body["cols"] == "placement"
    cells = body["cells"]
    total = client.get("/metrics").get_json()
    assert sum(c["revenue"] for c in cells) == pytest.approx(total["total_revenue"], abs=0.05)
    assert sum(c["record_count"] for c in cells) == total["total_records"]

    best = max(cells, key=lambda c: c["firm_margin"])
    assert (best["banner"], best["placement"]) == ("240 x 400", "ghi")


@needs_data
def test_pivot_accepts_aliases_and_filters(client):
    body = client.get("/pivot?rows=campaign&cols=engagement&placement=ghi").get_json()
    assert {c["campaign_number"] for c in body["cells"]} <= {"camp 1", "camp 2", "camp 3"}


@needs_data
@pytest.mark.parametrize("query", ["rows=banner&cols=banner", "rows=banner", "rows=date&cols=banner",
                                   "rows=banner&cols=channel"])
def test_pivot_rejects_bad_dimensions(client, query):
    assert client.get(f"/pivot?{query}").status_code == 400


# ---------------------------------------------------------------------
# Row data and CSV
# ---------------------------------------------------------------------

@needs_data
def test_data_pages_sorts_and_filters(client):
    body = client.get("/data?page=2&page_size=10&sort=revenue&order=desc&placement=ghi").get_json()
    assert body["page"] == 2 and body["page_size"] == 10 and len(body["rows"]) == 10
    assert body["pages"] == -(-body["total"] // 10)
    revenues = [r["revenue"] for r in body["rows"]]
    assert revenues == sorted(revenues, reverse=True)
    assert {r["placement"] for r in body["rows"]} == {"ghi"}
    assert body["columns"][0] == "date" and "firm_margin" in body["columns"]

    first = client.get("/data?page_size=10&sort=revenue&order=desc&placement=ghi").get_json()
    assert first["rows"][-1]["revenue"] >= revenues[0]


@needs_data
def test_data_clamps_page_and_size(client):
    body = client.get("/data?page=99999&page_size=100000").get_json()
    assert body["page_size"] == 500
    assert body["page"] == body["pages"]


@needs_data
@pytest.mark.parametrize("query", ["sort=password", "page=abc", "banner=nope"])
def test_data_rejects_bad_params(client, query):
    assert client.get(f"/data?{query}").status_code == 400


@needs_data
def test_data_csv_download(client):
    r = client.get("/data.csv?campaign_number=camp 2")
    assert r.status_code == 200
    assert r.mimetype == "text/csv"
    assert "attachment" in r.headers["Content-Disposition"]
    df = pd.read_csv(io.StringIO(r.get_data(as_text=True)))
    total = client.get("/metrics?campaign_number=camp 2").get_json()
    assert len(df) == total["total_records"]
    assert set(df["campaign_number"]) == {"camp 2"}
    assert df["revenue"].sum() == pytest.approx(total["total_revenue"], abs=0.05)


@needs_data
def test_api_info_lists_new_endpoints(client):
    endpoints = " ".join(client.get("/api/info").get_json()["endpoints"])
    for path in ("/pivot", "/data", "/data.csv", "/report/"):
        assert path in endpoints
