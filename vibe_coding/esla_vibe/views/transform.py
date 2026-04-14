"""
2D Linear Transformation Visualizer
Rewrite of trans.py — drag-free, fully browser-based.

Key insight: a 2×2 matrix is a function on R².
Shows the "before" grid (dashed) and the "after" grid (solid),
lets the user place points and watch them move, and optionally
shows eigenvectors and the determinant parallelogram.
"""
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from utils.linalg import grid_lines, apply_matrix_2d, eigeninfo, lerp_matrix
from utils.logger import get_logger

log = get_logger(__name__)

# ── Preset matrices ────────────────────────────────────────────────────────────
PRESETS = {
    "זהות (Identity)":        [[1, 0], [0, 1]],
    "סיבוב 45°":              [[0.707, -0.707], [0.707, 0.707]],
    "סיבוב 90°":              [[0, -1], [1, 0]],
    "סיבוב 180°":             [[-1, 0], [0, -1]],
    "שיקוף ציר X":            [[1, 0], [0, -1]],
    "שיקוף ציר Y":            [[-1, 0], [0, 1]],
    "שיקוף y=x":              [[0, 1], [1, 0]],
    "גזירה אופקית (Shear X)": [[1, 1], [0, 1]],
    "גזירה אנכית (Shear Y)":  [[1, 0], [1, 1]],
    "הגדלה 2×":               [[2, 0], [0, 2]],
    "דחיסה (Squeeze)":        [[2, 0], [0, 0.5]],
    "הטלה על X":              [[1, 0], [0, 0]],
    "הטלה על Y":              [[0, 0], [0, 1]],
}

GRAPH_ID = "transform-graph"


# ── Layout ─────────────────────────────────────────────────────────────────────
layout = html.Div([
    dbc.Row([
        # ── Left: the plane ───────────────────────────────────────────────────
        dbc.Col([
            dcc.Graph(
                id=GRAPH_ID,
                config={"displayModeBar": False, "scrollZoom": False},
                style={"height": "520px", "cursor": "crosshair"},
            ),
            # Animation slider
            html.Div([
                html.Small("t =", className="me-2 text-muted"),
                dcc.Slider(
                    id="transform-t", min=0, max=1, step=0.01, value=1,
                    marks={0: "I", 0.5: "½", 1: "T"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="flex-grow-1",
                ),
            ], className="d-flex align-items-center px-3 mt-1"),
            html.P(
                "גרור את הסליידר כדי לצפות בטרנספורמציה 'קורית'",
                className="text-center text-muted small mt-1",
            ),
        ], width=8),

        # ── Right: controls ───────────────────────────────────────────────────
        dbc.Col([
            html.H6("מטריצת הטרנספורמציה T", className="text-center fw-bold mb-2"),

            # 2×2 matrix input grid
            dbc.Row([
                dbc.Col(dbc.Input(id="t00", value="1", type="text",
                                  size="sm", className="text-center font-monospace"), width=6),
                dbc.Col(dbc.Input(id="t01", value="0", type="text",
                                  size="sm", className="text-center font-monospace"), width=6),
            ], className="mb-1 g-1"),
            dbc.Row([
                dbc.Col(dbc.Input(id="t10", value="0", type="text",
                                  size="sm", className="text-center font-monospace"), width=6),
                dbc.Col(dbc.Input(id="t11", value="1", type="text",
                                  size="sm", className="text-center font-monospace"), width=6),
            ], className="mb-2 g-1"),

            # Presets
            dbc.Select(
                id="transform-preset",
                options=[{"label": k, "value": k} for k in PRESETS],
                placeholder="בחר הגדרה מוכנה...",
                className="mb-3",
                size="sm",
            ),

            html.Hr(className="my-2"),

            # Options
            dbc.Label("אפשרויות תצוגה", className="small fw-bold"),
            dbc.Checklist(
                id="transform-options",
                options=[
                    {"label": " גריד מקורי (מקווקו)", "value": "orig_grid"},
                    {"label": " וקטורי בסיס e₁, e₂",  "value": "basis"},
                    {"label": " Eigenvectors",          "value": "eigen"},
                    {"label": " מקבילית det(T)",        "value": "det"},
                ],
                value=["orig_grid", "basis"],
                className="small mb-3",
            ),

            html.Hr(className="my-2"),

            # Stats card
            html.Div(id="transform-stats"),

            html.Hr(className="my-2"),

            # Point interaction
            html.P("לחץ על הגרף להוספת נקודה", className="text-muted small mb-1"),
            dbc.Button(
                "נקה נקודות", id="transform-clear",
                color="outline-secondary", size="sm",
            ),

            dcc.Store(id="transform-points", data=[]),
        ], width=4),
    ], className="g-2"),
], className="p-2")


# ── Figure builder ─────────────────────────────────────────────────────────────
def _build_fig(T_raw, t, options, points):
    T = lerp_matrix(T_raw, t)
    traces = []
    annotations = []

    # Invisible click-target layer — dense transparent markers cover the whole plot.
    # hoverinfo="none" (not "skip"!) is critical: "skip" disables click events too.
    _g = np.linspace(-5.2, 5.2, 22)
    _gx, _gy = np.meshgrid(_g, _g)
    traces.append(go.Scatter(
        x=_gx.flatten().tolist(), y=_gy.flatten().tolist(),
        mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=18),
        hoverinfo="none", showlegend=False,
    ))

    # Original grid (gray dashed)
    if "orig_grid" in options:
        for xs, ys in grid_lines([[1, 0], [0, 1]], bound=4, n=9):
            traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="rgba(120,120,120,0.25)", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

    # Transformed grid (blue)
    for xs, ys in grid_lines(T, bound=4, n=9):
        traces.append(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(99,179,237,0.35)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    T_arr = np.array(T, dtype=float)

    # Basis vectors
    if "basis" in options:
        for vec, color, label in [
            ([1, 0], "#f6ad55", "e₁"),
            ([0, 1], "#68d391", "e₂"),
        ]:
            img = (T_arr @ np.array(vec)).tolist()
            annotations.append(dict(
                x=img[0], y=img[1], ax=0, ay=0,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=3, arrowcolor=color, arrowwidth=3,
                text=f"T({label})", font=dict(color=color, size=11),
                showarrow=True,
            ))

    # Eigenvectors
    if "eigen" in options:
        info = eigeninfo(T_raw)   # always show true T eigenvectors
        for eg in info["pairs"]:
            v = np.array(eg["vec"]) * 4
            for sign in [1, -1]:
                annotations.append(dict(
                    x=float(v[0] * sign), y=float(v[1] * sign), ax=0, ay=0,
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=2, arrowcolor="#fc8181", arrowwidth=2,
                    text="", showarrow=True,
                ))

    # Determinant parallelogram
    if "det" in options:
        e1i = T_arr @ [1, 0]
        e2i = T_arr @ [0, 1]
        px = [0, e1i[0], e1i[0] + e2i[0], e2i[0], 0]
        py = [0, e1i[1], e1i[1] + e2i[1], e2i[1], 0]
        traces.append(go.Scatter(
            x=px, y=py, fill="toself",
            fillcolor="rgba(246,173,85,0.18)",
            line=dict(color="#f6ad55", width=1.5, dash="dash"),
            hoverinfo="skip", showlegend=False,
        ))

    # User points
    if points:
        orig_xs = [p[0] for p in points]
        orig_ys = [p[1] for p in points]
        trans = apply_matrix_2d(T, points)
        tx = [p[0] for p in trans]
        ty = [p[1] for p in trans]

        traces.append(go.Scatter(
            x=orig_xs, y=orig_ys, mode="markers",
            marker=dict(color="#f6ad55", size=9, symbol="circle-open",
                        line=dict(width=2, color="#f6ad55")),
            hoverinfo="skip", showlegend=False,
        ))
        traces.append(go.Scatter(
            x=tx, y=ty, mode="markers",
            marker=dict(color="#f6ad55", size=9),
            hoverinfo="skip", showlegend=False,
        ))
        for o, tr_ in zip(points, trans):
            if abs(o[0] - tr_[0]) > 0.02 or abs(o[1] - tr_[1]) > 0.02:
                annotations.append(dict(
                    x=tr_[0], y=tr_[1], ax=o[0], ay=o[1],
                    xref="x", yref="y", axref="x", ayref="y",
                    arrowhead=2, arrowcolor="rgba(246,173,85,0.55)",
                    arrowwidth=1.5, text="", showarrow=True,
                ))

    fig = go.Figure(traces)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        xaxis=dict(range=[-5.5, 5.5], showgrid=False, zeroline=True,
                   zerolinecolor="#374151", zerolinewidth=2,
                   fixedrange=True, color="#9ca3af"),
        yaxis=dict(range=[-5.5, 5.5], showgrid=False, zeroline=True,
                   zerolinecolor="#374151", zerolinewidth=2,
                   scaleanchor="x", fixedrange=True, color="#9ca3af"),
        margin=dict(l=30, r=10, t=10, b=30),
        annotations=annotations,
    )
    return fig


def _stats_card(T_raw):
    info = eigeninfo(np.array(T_raw, dtype=float))
    det = info["det"]
    det_color = "text-success" if abs(det) > 1e-4 else "text-danger"
    rows = [
        ("det(T)", f"{det:.4f}", det_color),
        ("trace(T)", f"{info['trace']:.4f}", ""),
        ("rank", str(info["rank"]), ""),
    ]
    for eg in info["pairs"]:
        v = eg["vec"]
        rows.append((f"λ = {eg['val']:.3f}",
                     f"({v[0]:.2f}, {v[1]:.2f})", "text-danger"))
    if not info["pairs"]:
        rows.append(("eigenvalues", "מרוכבים", "text-warning"))

    return dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.Small(label, className="text-muted"), width=5),
            dbc.Col(html.Small(val, className=f"font-monospace {cls}"), width=7),
        ]) for label, val, cls in rows
    ], className="p-2"))


# ── Callbacks ──────────────────────────────────────────────────────────────────
def _parse_matrix(t00, t01, t10, t11):
    def f(s):
        try:
            return float(eval(str(s)))  # allows "cos(0.5)", "1/2" etc.
        except Exception:
            return 0.0
    return [[f(t00), f(t01)], [f(t10), f(t11)]]


@callback(
    Output("t00", "value"), Output("t01", "value"),
    Output("t10", "value"), Output("t11", "value"),
    Input("transform-preset", "value"),
    prevent_initial_call=True,
)
def load_preset(name):
    if name not in PRESETS:
        return "1", "0", "0", "1"
    T = PRESETS[name]
    return (f"{T[0][0]:.4g}", f"{T[0][1]:.4g}",
            f"{T[1][0]:.4g}", f"{T[1][1]:.4g}")


@callback(
    Output(GRAPH_ID, "figure"),
    Output("transform-stats", "children"),
    Input("t00", "value"), Input("t01", "value"),
    Input("t10", "value"), Input("t11", "value"),
    Input("transform-t", "value"),
    Input("transform-options", "value"),
    Input("transform-points", "data"),
)
def update_figure(t00, t01, t10, t11, t, options, points):
    T = _parse_matrix(t00, t01, t10, t11)
    return _build_fig(T, t or 1, options or [], points or []), _stats_card(T)


@callback(
    Output("transform-points", "data"),
    Input(GRAPH_ID, "clickData"),
    Input("transform-clear", "n_clicks"),
    State("transform-points", "data"),
    prevent_initial_call=True,
)
def handle_click(click_data, _clear, points):
    if ctx.triggered_id == "transform-clear":
        log.info("points cleared")
        return []
    log.debug("clickData received: %s", click_data)
    if click_data:
        pt = click_data["points"][0]
        x, y = round(pt.get("x", 0), 3), round(pt.get("y", 0), 3)
        log.info("point added: (%.3f, %.3f)", x, y)
        return (points or []) + [[x, y]]
    log.warning("clickData fired but empty: %r", click_data)
    return points or []
