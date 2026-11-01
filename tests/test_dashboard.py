"""
Headless smoke tests for the Streamlit dashboard (streamlit.testing.v1.AppTest).

All tabs render in a single script run, so one run exercises every tab.
"""

from pathlib import Path

import pytest

from analysis.data_loader import DEFAULT_RAW_PATH
from models.predictor import model_available

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")

pytestmark = pytest.mark.skipif(not DEFAULT_RAW_PATH.exists(), reason="raw dataset not present")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_dashboard_renders_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "Online Advertising Performance"
    assert [m.label for m in app.metric][:3] == ["Revenue", "Placement cost", "Firm margin"]
    assert app.metric[0].value == "$276,263"


def test_dashboard_filter_updates_kpis(app):
    app.sidebar.multiselect[0].select("camp 2").run()
    assert not app.exception
    assert app.metric[0].value == "$34,890"   # camp 2 revenue
    app.sidebar.multiselect[0].unselect("camp 2").run()


@pytest.mark.skipif(not model_available(), reason="no trained model")
def test_dashboard_prediction_form(app):
    submit = next(b for b in app.button if b.label == "Predict")
    submit.click().run()
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert "Predicted revenue" in labels and "Predicted margin" in labels
