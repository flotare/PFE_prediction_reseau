import dash
from dash import State, dcc, html, Input, Output
import plotly.graph_objects as go

from collections import defaultdict

import graph
import utils

app = dash.Dash(__name__)

data = utils.get_data_day(12)

# récupérer les ids
user_ids = [d["meta"]["id"] for d in data]

# Préparer l'index des IDs par prefix (ici pour recherche par début)
prefix_index = defaultdict(list)
for uid in user_ids:
    s = str(uid)
    for i in range(1, min(len(s), 10)+1):  # on limite à 10 caractères max pour l’index
        prefix_index[s[:i]].append(uid)

# layout
app.layout = html.Div(
    [
        html.H1("Trajectoires utilisateurs"),
        dcc.Dropdown(
            id="user-dropdown",
            options=[
                {"label": str(uid), "value": uid} for uid in user_ids
            ],  # toutes les options dès le départ
            multi=True,
            placeholder="Tapez un ID utilisateur",
            searchable=True,
            value=[],  # valeur initiale vide
        ),
        dcc.Graph(id="trajectory-graph"),  # carte / trajectoire
        dcc.RadioItems(
            id="distance-mode",
            options=[
                {"label": "Brute", "value": "raw"},
                {"label": "Cumulée", "value": "cumulative"},
                {"label": "Depuis origine", "value": "origin"},
            ],
            value="raw",
            inline=True,
        ),
        dcc.Graph(id="distance-graph"),  # distances (affiché seulement 1 utilisateur)
    ]
)


@app.callback(
    Output("user-dropdown", "options"),
    Input("user-dropdown", "search_value"),
    State("user-dropdown", "value")
)
def update_dropdown_options(search_value, current_value):
    current_value = current_value or []

    if not search_value:
        # si rien tapé, on garde juste les IDs sélectionnés visibles
        return [{"label": str(uid), "value": uid} for uid in current_value]

    search_value = str(search_value)
    filtered = prefix_index.get(search_value[:10], [])  # on ne prend que max 10 premiers chars
    # garder seulement max 50 résultats pour ne pas saturer le Dropdown
    filtered = filtered[:50]

    # ajouter les IDs déjà sélectionnés pour qu'ils restent visibles
    filtered = list(set(filtered) | set(current_value))

    return [{"label": str(uid), "value": uid} for uid in filtered]


@app.callback(
    Output("trajectory-graph", "figure"),
    Output("distance-graph", "figure"),
    Output("distance-graph", "style"),
    Input("user-dropdown", "value"),
    Input("distance-mode", "value"),
)
def update_graph(selected_users, mode):
    # Aucun utilisateur sélectionné
    if not selected_users:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, {"display": "none"}

    # Un seul utilisateur
    if len(selected_users) == 1:
        user_id = int(selected_users[0])
        user = next(d for d in data if d["meta"]["id"] == user_id)

        traj_fig, dist_fig = graph.single_user_selected(user_id, user["events"], mode)

        return traj_fig, dist_fig, {"display": "block"}  # distance visible

    # Plusieurs utilisateurs
    traj_fig = graph.multiple_users_selected(selected_users, data)
    return traj_fig, go.Figure(), {"display": "none"}  # distance cachée


@app.callback(
    Output("distance-mode", "style"),
    Input("user-dropdown", "value"),
)
def toggle_distance_mode(selected_users):
    if selected_users is None:
        return {"display": "none"}
    if len(selected_users) == 1:
        return {"display": "block"}  # visible
    return {"display": "none"}  # caché


# run
if __name__ == "__main__":
    app.run(debug=True, dev_tools_hot_reload=True)

# http://127.0.0.1:8050
