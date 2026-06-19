from types import SimpleNamespace

from ui.admin import render_admin_tab
from ui.doctor import render_doctor_tab
from ui.patient import render_patient_tab
from ui.researcher import render_researcher_tab


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class FakeSt:
    def __init__(self):
        self.session_state = FakeSessionState()

    def markdown(self, *args, **kwargs):
        return None

    def segmented_control(self, *args, **kwargs):
        return kwargs.get("default")


def _deps(**handlers):
    calls = []

    def handler(name):
        def _inner(*args, **kwargs):
            calls.append((name, args, kwargs))
        return _inner

    defaults = {
        "st": FakeSt(),
        "render_patient_results": handler("patient_results"),
        "render_symptoms_tab": handler("symptoms"),
        "render_reminders": handler("reminders"),
        "render_patient_info": handler("patient_info"),
        "render_contact": handler("contact"),
        "render_feedback": handler("feedback"),
        "render_research_form": handler("research_form"),
        "render_doctor_eval_and_research": handler("doctor_eval_research"),
        "render_doctor_eval_form": handler("doctor_eval_form"),
        "render_general_info": handler("general_info"),
        "render_research_profile_team": handler("profile_team"),
        "render_researcher_analysis_video": handler("researcher_analysis"),
        "render_research_topic": handler("research_topic"),
        "render_research_info": handler("research_info"),
        "render_team": handler("team"),
        "render_admin_home": handler("admin_home"),
        "render_admin_management": handler("admin_management"),
        "render_change_password": handler("change_password"),
        "render_technology": handler("technology"),
    }
    defaults.update(handlers)
    return SimpleNamespace(calls=calls, **defaults)


def test_patient_results_tab_routes_to_patient_results():
    deps = _deps()

    render_patient_tab("📊 KẾT QUẢ ĐÁNH GIÁ", deps)

    assert deps.calls == [("patient_results", (), {})]


def test_doctor_management_tab_routes_to_combined_eval_and_research():
    deps = _deps()

    render_doctor_tab("📊 QUẢN LÝ ĐÁNH GIÁ & NCKH", deps)

    assert deps.calls == [("doctor_eval_form", (), {})]


def test_researcher_analysis_tab_routes_to_analysis_video():
    deps = _deps()

    render_researcher_tab("🔬 PHÂN TÍCH & TRÍCH XUẤT DỮ LIỆU", deps)

    assert deps.calls == [("researcher_analysis", (), {})]


def test_admin_management_tab_routes_to_admin_management():
    deps = _deps()

    render_admin_tab("🛠️ QUẢN TRỊ VIÊN", deps)

    assert deps.calls == [("admin_management", (), {})]
