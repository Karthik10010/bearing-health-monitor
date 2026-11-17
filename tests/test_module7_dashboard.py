"""Module 7: Streamlit AppTest smoke tests (top navigation, central input panel, every page)."""
import pytest

from pdm.config import PROJECT_ROOT, path

pytestmark = pytest.mark.skipif(not (path("models_dir") / "ppo_agent.zip").exists(), reason="train models first")
ENTRY = str(PROJECT_ROOT / "app" / "Home.py")
VIEWS = ["views/overview.py", "views/signal.py", "views/diagnosis.py", "views/maintenance.py",
         "views/assistant.py", "views/edge.py"]


@pytest.fixture(scope="module")
def report():
    from pdm.data.loader import load_upload
    from pdm.pipeline import analyze
    f = path("samples_dir") / "cwru_IR007_0.npy"
    return analyze(load_upload(f.read_bytes(), f.name), 12000, 1797, 0, source=f.name, whatif_rollouts=50)


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(ENTRY, default_timeout=300)  # generous: slow machines


@pytest.mark.parametrize("view", VIEWS)
def test_page_without_analysis(view):
    at = _app().run()
    at.switch_page(view).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("view", VIEWS)
def test_page_with_analysis(view, report):
    at = _app()
    at.session_state["report"] = report
    at.run()
    at.switch_page(view).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown) + " ".join(str(m.value) for m in at.metric)
    if view != "views/edge.py":
        assert report.diagnosis["fault"] in text  # status bar / results show the diagnosis


def test_analyze_button_runs_pipeline_and_inputs_persist():
    at = _app().run()
    at.button(key="analyze_btn").click().run()
    assert not at.exception, at.exception
    rep = at.session_state["report"]
    assert rep.diagnosis["fault"] in ("Normal", "Inner race", "Outer race", "Ball")
    # change an input, visit another page, come back: the value must survive
    at.number_input(key="w_rpm").set_value(1777.0).run()
    at.switch_page("views/assistant.py").run()
    at.switch_page("views/overview.py").run()
    assert not at.exception, at.exception
    assert at.number_input(key="w_rpm").value == 1777.0
    assert at.session_state["report"] is not None  # results are still there


def test_assistant_quick_question(report):
    at = _app()
    at.session_state["report"] = report
    at.run()
    at.switch_page("views/assistant.py").run()
    at.button(key="q0").click().run()
    assert not at.exception, at.exception
    assert len(at.session_state["chat"]) == 2


def test_gauge_needle_html():
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "app"))
    from components.common import gauge_html
    html = gauge_html(84, "Critical", "advanced damage")
    assert "rotate(61.2deg)" in html and "Critical" in html and "pdm-sweep" in html
