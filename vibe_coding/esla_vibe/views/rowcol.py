"""
Row / Column Vector Relationship
Rewrite of rawcol.py

Key insight: the row vectors of a 2×2 matrix [[a,b],[c,d]] are (a,b) and (c,d);
the column vectors are (a,c) and (b,d).
When row vectors become linearly dependent (parallel), so do column vectors —
and the determinant hits zero. This view makes that visible interactively.
"""
import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from utils.linalg import eigeninfo

_ARROW = dict(arrowhead=3, arrowwidth=3, showarrow=True, text="")


def _vec_fig(vecs, colors, labels, title):
    """Draw vectors as arrows from origin in a 2-D figure."""
    annotations = []
    for (x, y), color, label in zip(vecs, colors, labels):
        annotations.append(dict(
            x=x, y=y, ax=0, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            arrowcolor=color, arrowwidth=3, arrowhead=3,
            text=label, font=dict(color=color, size=13),
            showarrow=True,
        ))

    bound = max(max(abs(x) for x, _ in vecs + [(1, 0)]),
                max(abs(y) for _, y in vecs + [(0, 1)])) * 1.4 + 0.5

    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=13), x=0.5),
        showlegend=False,
        plot_bgcolor="#111827",
        paper_bgcolor="#1f2937",
        xaxis=dict(range=[-bound, bound], showgrid=True, gridcolor="#1f2937",
                   zeroline=True, zerolinecolor="#4b5563", zerolinewidth=2,
                   fixedrange=True, color="#9ca3af"),
        yaxis=dict(range=[-bound, bound], showgrid=True, gridcolor="#1f2937",
                   zeroline=True, zerolinecolor="#4b5563", zerolinewidth=2,
                   scaleanchor="x", fixedrange=True, color="#9ca3af"),
        margin=dict(l=30, r=10, t=40, b=30),
        annotations=annotations,
    )
    return fig


# ── Layout ─────────────────────────────────────────────────────────────────────
def _number_input(id_, val, label):
    return dbc.Col([
        html.Label(label, className="text-muted small"),
        dbc.Input(id=id_, value=str(val), type="number", step=0.1,
                  size="sm", className="text-center font-monospace"),
    ], width=3)


layout = html.Div([
    # Matrix controls
    dbc.Row([
        dbc.Col(html.H6("מטריצה A =", className="text-center pt-2"), width=2),
        _number_input("rc-a", 2,  "a"),
        _number_input("rc-b", 1,  "b"),
        _number_input("rc-c", 1,  "c"),
        _number_input("rc-d", 2,  "d"),
        dbc.Col(html.Div(id="rc-stats"), width=2),
    ], className="align-items-end mb-3 g-2"),

    # The two figures
    dbc.Row([
        dbc.Col([
            html.P(
                "וקטורי שורה: (a, b) ו-(c, d) — כחול",
                className="text-center small text-info",
            ),
            dcc.Graph(id="rc-row-fig",
                      config={"displayModeBar": False},
                      style={"height": "400px"}),
        ], width=6),
        dbc.Col([
            html.P(
                "וקטורי עמודה: (a, c) ו-(b, d) — כתום",
                className="text-center small text-warning",
            ),
            dcc.Graph(id="rc-col-fig",
                      config={"displayModeBar": False},
                      style={"height": "400px"}),
        ], width=6),
    ]),

    # Insight text
    dbc.Alert(id="rc-insight", color="secondary", className="mt-3 text-center"),
], className="p-2")


# ── Callbacks ──────────────────────────────────────────────────────────────────
@callback(
    Output("rc-row-fig", "figure"),
    Output("rc-col-fig", "figure"),
    Output("rc-stats", "children"),
    Output("rc-insight", "children"),
    Output("rc-insight", "color"),
    Input("rc-a", "value"), Input("rc-b", "value"),
    Input("rc-c", "value"), Input("rc-d", "value"),
)
def update_rowcol(a, b, c, d):
    def v(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    a, b, c, d = v(a), v(b), v(c), v(d)
    T = [[a, b], [c, d]]
    info = eigeninfo(np.array(T))
    det = info["det"]
    rank = info["rank"]

    row_fig = _vec_fig(
        [(a, b), (c, d)],
        ["#63b3ed", "#90cdf4"],
        ["r₁=(a,b)", "r₂=(c,d)"],
        "וקטורי שורה",
    )
    col_fig = _vec_fig(
        [(a, c), (b, d)],
        ["#f6ad55", "#fbd38d"],
        ["c₁=(a,c)", "c₂=(b,d)"],
        "וקטורי עמודה",
    )

    stats = dbc.Card(dbc.CardBody([
        html.Div(f"det = {det:.3f}", className="font-monospace small"),
        html.Div(f"rank = {rank}",   className="font-monospace small"),
    ], className="p-1 text-center"))

    if abs(det) < 1e-8:
        insight = "⚠️  הדטרמיננטה = 0: שני וקטורי השורה תלויים לינארית — וכך גם שני וקטורי העמודה. הצורה שטוחה."
        color = "danger"
    elif rank == 2:
        insight = "✓  הדטרמיננטה ≠ 0: וקטורי השורה בלתי-תלויים לינארית — וכך גם וקטורי העמודה. המטריצה הפיכה."
        color = "success"
    else:
        insight = f"rank = {rank}"
        color = "warning"

    return row_fig, col_fig, stats, insight, color
