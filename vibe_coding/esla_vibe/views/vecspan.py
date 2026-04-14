"""
3-D Vector Span Visualizer
Rewrite of vec_comb.py (originally VPython)

Up to 4 vectors in R³. Toggle them on/off.
Shows the span:
  • 1 independent vector  → line through origin
  • 2 independent vectors → plane through origin
  • 3 independent vectors → all of R³

Vectors are displayed as 3-D cones (arrows from origin).
"""
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from utils.linalg import span_geometry

_N_VECS = 4
_COLORS = ["#63b3ed", "#f6ad55", "#68d391", "#fc8181"]
_LABELS = ["v₁", "v₂", "v₃", "v₄"]

# Default initial vectors
_DEFAULTS = [
    [2, 0, 0],
    [0, 2, 0],
    [1, 1, 2],
    [-1, 1, 0],
]


# ── Layout helpers ─────────────────────────────────────────────────────────────
def _vec_controls(i):
    vi = _DEFAULTS[i]
    color = _COLORS[i]
    label = _LABELS[i]
    return dbc.Card([
        dbc.CardHeader(
            dbc.Row([
                dbc.Col(html.Span(label, style={"color": color, "fontWeight": "bold"}),
                        width="auto"),
                dbc.Col(
                    dbc.Switch(id=f"vec-on-{i}", value=True, className="float-end"),
                    width="auto", className="ms-auto",
                ),
            ], className="g-0 align-items-center"),
            className="py-1 px-2",
        ),
        dbc.CardBody([
            *[html.Div([
                html.Small(ax, className="text-muted me-1"),
                dcc.Slider(
                    id=f"vec-{i}-{j}", min=-3, max=3, step=0.25,
                    value=vi[j],
                    marks={-3: "-3", 0: "0", 3: "3"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="mb-1",
                ),
            ]) for j, ax in enumerate(["x", "y", "z"])]
        ], className="py-2 px-2"),
    ], className="mb-2", style={"background": "#1f2937", "border": "1px solid #374151"})


layout = html.Div([
    dbc.Row([
        # ── 3-D graph ──────────────────────────────────────────────────────
        dbc.Col([
            dcc.Graph(id="span-graph",
                      config={"displayModeBar": True},
                      style={"height": "560px"}),
            dbc.Alert(id="span-info", color="secondary",
                      className="text-center small py-2 mt-1"),
        ], width=8),

        # ── Vector controls ────────────────────────────────────────────────
        dbc.Col([
            html.H6("וקטורים", className="text-center fw-bold mb-2"),
            *[_vec_controls(i) for i in range(_N_VECS)],
        ], width=4, style={"maxHeight": "580px", "overflowY": "auto"}),
    ]),
], className="p-2")


# ── Figure builder ─────────────────────────────────────────────────────────────
def _build_fig(vectors, active):
    traces = []

    # Axes
    for axis, color in [([1, 0, 0], "#fc8181"),
                         ([0, 1, 0], "#68d391"),
                         ([0, 0, 1], "#63b3ed")]:
        traces.append(go.Scatter3d(
            x=[0, axis[0] * 3], y=[0, axis[1] * 3], z=[0, axis[2] * 3],
            mode="lines",
            line=dict(color=color, width=2),
            hoverinfo="skip", showlegend=False,
        ))

    active_vecs = [v for v, on in zip(vectors, active) if on]

    # Span surface / line
    kind, data = span_geometry(active_vecs)
    if kind == "line":
        traces.append(go.Scatter3d(
            x=data["x"], y=data["y"], z=data["z"],
            mode="lines",
            line=dict(color="rgba(246,173,85,0.35)", width=4),
            hoverinfo="skip", showlegend=False,
        ))
    elif kind == "plane":
        xs = data["x"]
        n = int(len(xs) ** 0.5)
        x2d = [xs[i * n:(i + 1) * n] for i in range(n)]
        y2d = [data["y"][i * n:(i + 1) * n] for i in range(n)]
        z2d = [data["z"][i * n:(i + 1) * n] for i in range(n)]
        traces.append(go.Surface(
            x=x2d, y=y2d, z=z2d,
            colorscale=[[0, "rgba(99,179,237,0.15)"],
                        [1, "rgba(99,179,237,0.25)"]],
            showscale=False, hoverinfo="skip",
        ))

    # Vectors as cones
    for i, (vec, on, color, label) in enumerate(
            zip(vectors, active, _COLORS, _LABELS)):
        if not on:
            continue
        v = np.array(vec, dtype=float)
        traces.append(go.Cone(
            x=[0], y=[0], z=[0],
            u=[v[0]], v=[v[1]], w=[v[2]],
            colorscale=[[0, color], [1, color]],
            showscale=False, sizemode="absolute", sizeref=0.3,
            hovertemplate=f"{label} = ({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})<extra></extra>",
        ))
        # Shaft
        traces.append(go.Scatter3d(
            x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode="lines+text",
            line=dict(color=color, width=4),
            text=["", label],
            textfont=dict(color=color, size=14),
            textposition="top center",
            hoverinfo="skip", showlegend=False,
        ))

    fig = go.Figure(traces)
    bound = 3.5
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="#111827",
        scene=dict(
            xaxis=dict(range=[-bound, bound], backgroundcolor="#111827",
                       gridcolor="#1f2937", showbackground=True, color="#9ca3af"),
            yaxis=dict(range=[-bound, bound], backgroundcolor="#111827",
                       gridcolor="#1f2937", showbackground=True, color="#9ca3af"),
            zaxis=dict(range=[-bound, bound], backgroundcolor="#111827",
                       gridcolor="#1f2937", showbackground=True, color="#9ca3af"),
            bgcolor="#111827",
            camera=dict(eye=dict(x=1.4, y=1.4, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


def _span_info(vectors, active):
    active_vecs = [v for v, on in zip(vectors, active) if on]
    kind, _ = span_geometry(active_vecs)
    n_active = sum(active)

    if n_active == 0:
        return "אין וקטורים פעילים", "secondary"

    # Compute actual rank
    if active_vecs:
        rank = int(np.linalg.matrix_rank(np.array(active_vecs, dtype=float)))
    else:
        rank = 0

    texts = {
        0: ("קבוצה ריקה", "secondary"),
        1: (f"ה-span הוא קו דרך הראשית (מימד 1)", "info"),
        2: (f"ה-span הוא מישור דרך הראשית (מימד 2)", "primary"),
        3: (f"ה-span הוא כל R³ (מימד 3)", "success"),
    }
    msg, color = texts.get(rank, ("", "secondary"))
    if rank < n_active:
        msg += f" — שימו לב: {n_active - rank} וקטור{'ים' if n_active - rank > 1 else ''} תלוי{'ים' if n_active - rank > 1 else ''}"
        color = "warning"
    return msg, color


# ── Callbacks ──────────────────────────────────────────────────────────────────
_vec_inputs = (
    [Input(f"vec-{i}-{j}", "value") for i in range(_N_VECS) for j in range(3)] +
    [Input(f"vec-on-{i}", "value") for i in range(_N_VECS)]
)


@callback(
    Output("span-graph", "figure"),
    Output("span-info",  "children"),
    Output("span-info",  "color"),
    *_vec_inputs,
)
def update_span(*args):
    coords = list(args[:_N_VECS * 3])
    active = list(args[_N_VECS * 3:])

    vectors = []
    for i in range(_N_VECS):
        x = coords[i * 3 + 0] or 0
        y = coords[i * 3 + 1] or 0
        z = coords[i * 3 + 2] or 0
        vectors.append([float(x), float(y), float(z)])

    active = [bool(a) for a in active]
    fig = _build_fig(vectors, active)
    msg, color = _span_info(vectors, active)
    return fig, msg, color
