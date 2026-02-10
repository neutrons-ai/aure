"""Flask blueprint – page routes and JSON API endpoints."""

from flask import (
    Blueprint,
    current_app,
    jsonify,
    redirect,
    render_template,
    url_for,
)

from .data import RunData

bp = Blueprint(
    "web",
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static/web",
)


def _run_data() -> RunData:
    return RunData(current_app.config["OUTPUT_DIR"])


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------


@bp.route("/")
def index():
    return redirect(url_for("web.history"))


@bp.route("/history")
def history():
    rd = _run_data()
    return render_template(
        "history.html",
        run_info=rd.get_run_info(),
        active_tab="history",
    )


@bp.route("/results")
def results():
    rd = _run_data()
    return render_template(
        "results.html",
        run_info=rd.get_run_info(),
        active_tab="results",
    )


# ------------------------------------------------------------------
# JSON API endpoints  (consumed by Plotly.js on the client)
# ------------------------------------------------------------------


@bp.route("/api/run-info")
def api_run_info():
    return jsonify(_run_data().get_run_info())


@bp.route("/api/chi2")
def api_chi2():
    return jsonify(_run_data().get_chi2_progression())


@bp.route("/api/reflectivity")
def api_reflectivity():
    return jsonify(_run_data().get_reflectivity_data())


@bp.route("/api/sld")
def api_sld():
    return jsonify(_run_data().get_sld_profiles())


@bp.route("/api/parameters")
def api_parameters():
    return jsonify(_run_data().get_fit_parameters())
