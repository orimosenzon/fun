"""
Gauss-Jordan Elimination — step-by-step visualizer
Rewrite of gEliminate.py

Enter any m×n augmented matrix (supports fractions).
Navigate through each elementary row operation with full explanation.
Pivot rows and columns are highlighted.
"""
from dash import html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc

from utils.linalg import gaussian_steps
from utils.logger import get_logger

log = get_logger(__name__)

# ── Layout ─────────────────────────────────────────────────────────────────────
_EXAMPLE_MATRICES = {
    "2×3: מערכת פשוטה":   "2 1 5\n4 3 11",
    "3×4: מערכת 3 משוואות": "1 2 -1 8\n2 1 0 5\n-1 3 2 7",
    "3×3: מטריצה סינגולרית": "1 2 3\n4 5 6\n7 8 9",
    "4×5: מערכת מלאה":     "2 -1 0 0 1\n-1 2 -1 0 0\n0 -1 2 -1 0\n0 0 -1 2 1",
}

layout = html.Div([
    dbc.Row([
        # ── Input panel ──────────────────────────────────────────────────────
        dbc.Col([
            html.H6("מטריצה מוגדלת", className="fw-bold mb-1"),
            html.Small("שורה לכל שורה, מרווחים בין מספרים. תומך בשברים: 1/2",
                       className="text-muted"),
            dbc.Textarea(
                id="gauss-input",
                value="2 1 5\n4 3 11",
                style={"fontFamily": "monospace", "direction": "ltr",
                       "height": "130px", "background": "#1f2937",
                       "color": "#e2e8f0", "border": "1px solid #374151"},
                className="mt-1 mb-2",
            ),
            dbc.Select(
                id="gauss-example",
                options=[{"label": k, "value": k} for k in _EXAMPLE_MATRICES],
                placeholder="טען דוגמה...",
                size="sm", className="mb-2",
            ),
            dbc.Button("פתור", id="gauss-solve", color="primary",
                       size="sm", className="me-2"),
            dbc.Button("אפס", id="gauss-reset", color="outline-secondary",
                       size="sm"),
        ], width=4),

        # ── Step display ─────────────────────────────────────────────────────
        dbc.Col([
            # Navigation
            dbc.Row([
                dbc.Col(dbc.Button("◀ הקודם", id="gauss-prev",
                                   color="outline-secondary", size="sm",
                                   disabled=True), width="auto"),
                dbc.Col(html.Div(id="gauss-step-label",
                                 className="text-center text-muted small pt-1"),
                        className="flex-grow-1"),
                dbc.Col(dbc.Button("הבא ▶", id="gauss-next",
                                   color="outline-primary", size="sm",
                                   disabled=True), width="auto"),
            ], className="align-items-center mb-2"),

            # Operation description
            dbc.Alert(id="gauss-op", color="info",
                      className="text-center font-monospace py-2 mb-2 small"),

            # Matrix display
            html.Div(id="gauss-matrix-display"),
        ], width=8),
    ]),

    # Hidden stores
    dcc.Store(id="gauss-steps", data=[]),
    dcc.Store(id="gauss-idx",   data=0),
], className="p-2")


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _parse_matrix(text):
    """Parse textarea into list of lists of numbers (as strings for sympy)."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        nums = line.split()
        rows.append(nums)
    if not rows:
        raise ValueError("ריק")
    ncols = len(rows[0])
    for r in rows:
        if len(r) != ncols:
            raise ValueError("שורות באורכים שונים")
    return rows


def _render_matrix(step):
    """Render the matrix as an HTML table with highlighted pivot row/col."""
    mat = step["matrix"]
    pivot_row = step.get("pivot_row")
    pivot_col = step.get("pivot_col")
    n_cols = len(mat[0]) if mat else 0

    header_cells = []

    tbody_rows = []
    for ri, row in enumerate(mat):
        cells = []
        for ci, val in enumerate(row):
            is_pivot_row = (ri == pivot_row)
            is_pivot_col = (ci == pivot_col)
            is_aug_sep = (ci == n_cols - 2)  # right before last col = separator

            style = {"padding": "6px 12px", "textAlign": "center",
                     "fontFamily": "monospace", "fontSize": "1rem"}
            if is_pivot_row and is_pivot_col:
                style["background"] = "#1e3a5f"
                style["color"] = "#63b3ed"
                style["fontWeight"] = "bold"
            elif is_pivot_row:
                style["background"] = "#1a2d40"
                style["color"] = "#90cdf4"
            elif is_pivot_col:
                style["color"] = "#68d391"

            border_right = "2px solid #4b5563" if is_aug_sep else None
            if border_right:
                style["borderRight"] = border_right

            cells.append(html.Td(val, style=style))

        row_style = {}
        if ri == pivot_row:
            row_style = {"background": "rgba(49,130,206,0.08)"}
        tbody_rows.append(html.Tr(cells, style=row_style))

    return html.Div(
        html.Table(
            [html.Tbody(tbody_rows)],
            style={
                "borderCollapse": "collapse",
                "background": "#111827",
                "border": "1px solid #374151",
                "borderRadius": "6px",
                "overflow": "hidden",
                "margin": "0 auto",
            }
        ),
        className="text-center"
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────
@callback(
    Output("gauss-input", "value"),
    Input("gauss-example", "value"),
    prevent_initial_call=True,
)
def load_example(name):
    return _EXAMPLE_MATRICES.get(name, "")


@callback(
    Output("gauss-steps", "data"),
    Output("gauss-idx",   "data"),
    Input("gauss-solve",  "n_clicks"),
    Input("gauss-reset",  "n_clicks"),
    State("gauss-input",  "value"),
    prevent_initial_call=True,
)
def solve(_, __, text):
    if ctx.triggered_id == "gauss-reset":
        log.info("gaussian reset")
        return [], 0
    try:
        rows = _parse_matrix(text or "")
        log.info("gaussian solve: %dx%d matrix", len(rows), len(rows[0]) if rows else 0)
        steps = gaussian_steps(rows)
        log.debug("produced %d steps", len(steps))
        return steps, 0
    except Exception as e:
        log.error("gaussian solve failed: %s", e, exc_info=True)
        return [{"matrix": [], "op": f"שגיאה: {e}",
                 "pivot_row": None, "pivot_col": None}], 0


@callback(
    Output("gauss-idx", "data", allow_duplicate=True),
    Input("gauss-prev", "n_clicks"),
    Input("gauss-next", "n_clicks"),
    State("gauss-idx",  "data"),
    State("gauss-steps", "data"),
    prevent_initial_call=True,
)
def navigate(prev_clicks, next_clicks, idx, steps):
    n = len(steps)
    if not n:
        return 0
    if ctx.triggered_id == "gauss-prev":
        return max(0, idx - 1)
    return min(n - 1, idx + 1)


@callback(
    Output("gauss-matrix-display", "children"),
    Output("gauss-op",             "children"),
    Output("gauss-step-label",     "children"),
    Output("gauss-prev",           "disabled"),
    Output("gauss-next",           "disabled"),
    Input("gauss-steps", "data"),
    Input("gauss-idx",   "data"),
)
def render_step(steps, idx):
    if not steps:
        empty = html.P("הכנס מטריצה ולחץ על 'פתור'",
                       className="text-muted text-center mt-4")
        return empty, "ממתין לקלט...", "", True, True

    step = steps[idx]
    n = len(steps)
    mat_display = _render_matrix(step) if step["matrix"] else html.P("שגיאה")
    label = f"שלב {idx + 1} מתוך {n}"
    op = step.get("op", "")
    return mat_display, op, label, (idx == 0), (idx == n - 1)
