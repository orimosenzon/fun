"""
esla_vibe — Linear Algebra Visualization Suite
Rewrite of https://github.com/orimosenzon/fun/tree/master/esla

Run:
    python app.py
Then open http://localhost:5050
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Import all views (registers their callbacks)
from views import transform, rowcol, gaussian, vecspan, surface3d
from utils.logger import get_logger

log = get_logger(__name__)

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title="אלגברה לינארית",
)

# ── Navbar ─────────────────────────────────────────────────────────────────────
_NAV_TABS = [
    ("🔀 טרנספורמציות", "transform"),
    ("↔ שורות ועמודות", "rowcol"),
    ("∑ אלימינציה גאוסית", "gaussian"),
    ("⟂ מרחב פריסה 3D", "vecspan"),
    ("∫ פונקציות 3D", "surface3d"),
]

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink(
            label, id=f"nav-{tab_id}", href=f"#{tab_id}",
            className="nav-tab-link",
        ))
        for label, tab_id in _NAV_TABS
    ],
    brand=html.Span([
        html.Span("esla", style={"color": "#f0c040", "fontWeight": "bold"}),
        html.Span("_vibe", style={"color": "#63b3ed"}),
        html.Span(" | אלגברה לינארית", style={"color": "#9ca3af", "fontSize": "0.85em"}),
    ]),
    brand_href="#",
    color="dark",
    dark=True,
    className="mb-0 shadow-sm",
    fluid=True,
)

# ── Main layout ────────────────────────────────────────────────────────────────
_TABS = [
    dcc.Tab(
        label=label, value=tab_id,
        className="custom-tab", selected_className="custom-tab--selected",
        children=view.layout,
    )
    for (label, tab_id), view in zip(_NAV_TABS, [transform, rowcol, gaussian, vecspan, surface3d])
]

app.layout = html.Div([
    navbar,
    dcc.Tabs(
        id="main-tabs",
        value="transform",
        children=_TABS,
        className="mt-0",
        colors={"border": "#374151", "primary": "#f0c040", "background": "#1f2937"},
    ),
], style={"background": "#111827", "minHeight": "100vh"})


if __name__ == "__main__":
    log.info("esla_vibe starting on port 5050")
    app.run(debug=True, port=5050)
