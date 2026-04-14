"""
3-D Function Surface Visualizer
Rewrite of 3dFunction.py (originally Tkinter + manual rotation).

Enter any f(x,y) expression; see it as an interactive 3-D surface.
Plotly provides free 3-D rotation/zoom.
"""
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from utils.linalg import eval_surface

_EXAMPLES = {
    "sin(x)·sin(y) (המקור)":   "sin(x)*sin(y)",
    "אוכף x²−y²":               "x**2 - y**2",
    "פרבולואיד x²+y²":          "x**2 + y**2",
    "Gaussian":                  "exp(-(x**2+y**2)/4)",
    "גלים sin(sqrt(x²+y²))":    "sin(sqrt(x**2+y**2+1e-6))",
    "שטח מנוחה sin(x)·cos(y)":  "sin(x)*cos(y)",
    "המישור 0":                  "0",
}

layout = html.Div([
    dbc.Row([
        # Graph
        dbc.Col([
            dcc.Graph(id="surf-graph",
                      config={"displayModeBar": True},
                      style={"height": "540px"}),
        ], width=8),

        # Controls
        dbc.Col([
            html.H6("פונקציה f(x, y)", className="fw-bold mb-1"),
            dbc.Input(
                id="surf-expr", value="sin(x)*sin(y)",
                type="text", placeholder="sin(x)*cos(y)",
                className="font-monospace mb-2",
                style={"direction": "ltr"},
            ),
            html.Small(
                "תומך ב: sin, cos, tan, exp, log, sqrt, abs, pi, e, x**2, ...",
                className="text-muted",
            ),

            html.Hr(className="my-2"),

            dbc.Select(
                id="surf-example",
                options=[{"label": k, "value": k} for k in _EXAMPLES],
                placeholder="דוגמאות...",
                size="sm", className="mb-3",
            ),

            dbc.Label("טווח (−N עד N)", className="small"),
            dcc.Slider(
                id="surf-bound", min=2, max=10, step=1, value=4,
                marks={i: str(i) for i in [2, 4, 6, 8, 10]},
                className="mb-3",
            ),

            dbc.Label("רזולוציה", className="small"),
            dcc.Slider(
                id="surf-res", min=20, max=100, step=10, value=60,
                marks={20: "נמוכה", 60: "רגילה", 100: "גבוהה"},
                className="mb-3",
            ),

            dbc.Select(
                id="surf-colorscale",
                options=[
                    {"label": cs, "value": cs}
                    for cs in ["Viridis", "Plasma", "RdBu", "Turbo", "Cividis"]
                ],
                value="Viridis",
                size="sm", className="mb-3",
            ),

            dbc.Alert(id="surf-error", color="danger",
                      className="small py-1 d-none"),
        ], width=4),
    ]),
], className="p-2")


@callback(
    Output("surf-expr", "value"),
    Input("surf-example", "value"),
    prevent_initial_call=True,
)
def load_example(name):
    return _EXAMPLES.get(name, "")


@callback(
    Output("surf-graph",  "figure"),
    Output("surf-error",  "children"),
    Output("surf-error",  "className"),
    Input("surf-expr",       "value"),
    Input("surf-bound",      "value"),
    Input("surf-res",        "value"),
    Input("surf-colorscale", "value"),
)
def update_surface(expr, bound, res, colorscale):
    try:
        X, Y, Z = eval_surface(expr or "0", bound=bound or 4, n=res or 60)
    except Exception as e:
        empty = go.Figure()
        empty.update_layout(paper_bgcolor="#111827", plot_bgcolor="#111827")
        return empty, f"שגיאה בנוסחה: {e}", "small py-1"

    fig = go.Figure(go.Surface(
        x=X, y=Y, z=Z,
        colorscale=colorscale,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="#f6ad55",
                   project_z=False, width=1),
        ),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#111827",
        scene=dict(
            xaxis=dict(title="x", backgroundcolor="#111827",
                       gridcolor="#1f2937", showbackground=True, color="#9ca3af"),
            yaxis=dict(title="y", backgroundcolor="#111827",
                       gridcolor="#1f2937", showbackground=True, color="#9ca3af"),
            zaxis=dict(title=f"f(x,y)", backgroundcolor="#111827",
                       gridcolor="#1f2937", showbackground=True, color="#9ca3af"),
            bgcolor="#111827",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig, "", "d-none"
