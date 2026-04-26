import plotly.express as px
import plotly.graph_objects as go

import utils

import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def plot_trajectory(traj, user_id=None, multiple=False):
    """
    traj : dict de arrays, retourné par build_trajectory
    keys : "latitude", "longitude", "time", etc.
    """

    fig = go.Figure()

    lat = traj["latitude"]
    lon = traj["longitude"]
    time = traj["time"]

    if multiple:
        # Plusieurs utilisateurs
        fig.add_trace(
            go.Scatter(
                x=lat,
                y=lon,
                mode="lines+markers",
                name=f"user {user_id}" if user_id else "trajet",
            )
        )
    else:
        # Un seul utilisateur
        fig = px.scatter(
            x=lat,
            y=lon,
            animation_frame=time,
            title=f"Trajectoire utilisateur {user_id}",
        )
        fig.add_scatter(
            x=lat,
            y=lon,
            mode="lines",
            name="trajet",
        )

    lat_margin = (np.max(lat) - np.min(lat)) * 0.05
    lon_margin = (np.max(lon) - np.min(lon)) * 0.05

    fig.update_xaxes(range=[np.min(lat) - lat_margin, np.max(lat) + lat_margin])
    fig.update_yaxes(range=[np.min(lon) - lon_margin, np.max(lon) + lon_margin])

    return fig


def plot_distance(traj_df, mode="raw"):
    if mode == "raw":
        y_col = "dist_raw"
        title = "Distance brute"
    elif mode == "cumulative":
        y_col = "dist_cumulative"
        title = "Distance cumulée"
    else:
        y_col = "dist_origin"
        title = "Distance depuis origine"

    fig = px.line(
        traj_df,
        x="time",
        y=y_col,
        markers=True,
        title=title,
    )

    fig.update_yaxes(title=f"Distance (m)")
    return fig


def single_user_selected(user_id, events, mode="raw"):
    traj_df = utils.build_trajectory(events)
    fig_traj = plot_trajectory(traj_df, user_id=user_id)
    fig_dist = plot_distance(traj_df, mode=mode)
    return fig_traj, fig_dist


def multiple_users_selected(selected_users, events):
    fig = go.Figure()

    # Ajout des antennes
    fig.add_trace(
        go.Scatter(
            x=utils.coords[:, 0],
            y=utils.coords[:, 1],
            mode="markers",
            marker=dict(size=3, color="lightgray"),
            name="antennes",
        )
    )

    for user_id in selected_users:
        user_id = int(user_id)
        events_user = events[user_id]
        if events_user is None:
            continue

        traj_df = utils.build_trajectory(events_user)

        fig.add_trace(
            go.Scatter(
                x=traj_df["latitude"],
                y=traj_df["longitude"],
                mode="lines+markers",
                name=f"user {user_id}",
            )
        )
    fig.update_layout(
        title="Trajectoires utilisateurs sélectionnés",
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        legend_title="Utilisateurs",
    )

    return fig
