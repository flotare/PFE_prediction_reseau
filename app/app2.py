import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from collections import defaultdict
from functools import lru_cache

import utils

# ======================
# APP INIT
# ======================
app = dash.Dash(__name__)

AVAILABLE_DAYS = [12, 19]


# ======================
# DATA LOADING (CACHED)
# ======================
@lru_cache(maxsize=5)
def load_day(day: int):
    events, metas = utils.get_pickle_day(day)

    prefix_index = defaultdict(set)
    for uid in metas.keys():
        s = str(uid)
        for i in range(1, min(len(s), 10) + 1):
            prefix_index[s[:i]].add(uid)

    return events, metas, prefix_index


# ======================
# LAYOUT
# ======================
app.layout = html.Div([

    html.H1("Comparaison de trajectoires utilisateurs"),

    html.Div([
        html.Div([
            html.H3("Utilisateur A"),
            dcc.Dropdown(
                id="day-a",
                options=[{"label": f"Jour {d}", "value": d} for d in AVAILABLE_DAYS],
                value=AVAILABLE_DAYS[0],
                clearable=False,
            ),
            dcc.Dropdown(
                id="user-a",
                placeholder="Rechercher user A",
            ),
        ], style={"width": "48%", "display": "inline-block"}),

        html.Div([
            html.H3("Utilisateur B"),
            dcc.Dropdown(
                id="day-b",
                options=[{"label": f"Jour {d}", "value": d} for d in AVAILABLE_DAYS],
                value=AVAILABLE_DAYS[-1],
                clearable=False,
            ),
            dcc.Dropdown(
                id="user-b",
                placeholder="Rechercher user B",
            ),
        ], style={"width": "48%", "display": "inline-block", "marginLeft": "4%"}),
    ]),

    html.Br(),

    dcc.Graph(id="trajectory-graph"),
    dcc.Graph(id="distance-graph"),
])


# ======================
# HELPERS
# ======================
def search_users(search_value: str, prefix_index: dict) -> list[dict]:
    if not search_value:
        return []
    matches = list(prefix_index.get(str(search_value)[:10], []))[:50]
    return [{"label": str(uid), "value": uid} for uid in sorted(matches)]


def get_user_events(events: dict, user_id):
    """Robuste au type int/str des clés."""
    if user_id in events:
        return events[user_id]
    try:
        return events[int(user_id)]
    except (KeyError, ValueError, TypeError):
        pass
    try:
        return events[str(user_id)]
    except KeyError:
        return None


# ======================
# DROPDOWNS
# ======================
@app.callback(
    Output("user-a", "options"),
    Input("user-a", "search_value"),
    Input("day-a", "value"),
    State("user-a", "options"),
    prevent_initial_call=True,
)
def update_user_a(search_value, day, current_options):
    _, _, prefix_index = load_day(day)
    if not search_value:
        return current_options or []
    return search_users(search_value, prefix_index)


@app.callback(
    Output("user-b", "options"),
    Input("user-b", "search_value"),
    Input("day-b", "value"),
    State("user-b", "options"),
    prevent_initial_call=True,
)
def update_user_b(search_value, day, current_options):
    _, _, prefix_index = load_day(day)
    if not search_value:
        return current_options or []
    return search_users(search_value, prefix_index)


# ======================
# MAIN CALLBACK
# ======================
@app.callback(
    Output("trajectory-graph", "figure"),
    Output("distance-graph", "figure"),
    Input("user-a", "value"),
    Input("day-a", "value"),
    Input("user-b", "value"),
    Input("day-b", "value"),
)
def compare_users(user_a, day_a, user_b, day_b):
    empty = go.Figure(), go.Figure()

    if not user_a or not user_b:
        return empty

    events_a, _, _ = load_day(day_a)
    events_b, _, _ = load_day(day_b)

    ev_a = get_user_events(events_a, user_a)
    ev_b = get_user_events(events_b, user_b)

    if ev_a is None or ev_b is None:
        return empty

    # Trajectoires
    traj_a = utils.build_trajectory(ev_a)
    traj_b = utils.build_trajectory(ev_b)

    if len(traj_a["longitude"]) == 0 or len(traj_b["longitude"]) == 0:
        return empty

    # Distance
    dist = utils.compute_merge_on_timeline_distance(ev_a, ev_b)

    # Figure trajectoires  (lon = x, lat = y — convention cartographique)
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(
        x=traj_a["longitude"],
        y=traj_a["latitude"],
        mode="lines+markers",
        name=f"A — user {user_a}, jour {day_a}",
    ))
    fig_traj.add_trace(go.Scatter(
        x=traj_b["longitude"],
        y=traj_b["latitude"],
        mode="lines+markers",
        name=f"B — user {user_b}, jour {day_b}",
    ))
    fig_traj.update_layout(
        title="Trajectoires",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
    )

    # Figure distance
    fig_dist = go.Figure()
    fig_dist.add_annotation(
        text=f"Distance temporelle = {dist:,.1f} m·s",
        showarrow=False,
        font=dict(size=22),
        x=0.5, y=0.5, xref="paper", yref="paper",
    )
    fig_dist.update_layout(
        title="Similarité de trajectoire",
        xaxis={"visible": False},
        yaxis={"visible": False},
    )

    return fig_traj, fig_dist


# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(debug=True)