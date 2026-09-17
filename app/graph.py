import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import utils

import numpy as np

import imageio.v2 as imageio
import plotly.io as pio
import tempfile
import os

from datetime import datetime
from collections import Counter

import pandas as pd

def plot_antennas(csv_file):
    # Lecture du fichier
    df = pd.read_csv(csv_file)

    # Création de la figure
    fig = go.Figure()

    fig.add_trace(
        go.Scattermapbox(
            lat=df["lat"],
            lon=df["lon"],
            mode="markers",
            marker=dict(
                size=8,
                color="red",
                opacity=0.8,
            ),
            text=df["cellid"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latitude : %{lat:.6f}<br>"
                "Longitude : %{lon:.6f}<extra></extra>"
            ),
        )
    )

    # Centre de la carte
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=df["lat"].mean(),
                lon=df["lon"].mean(),
            ),
            zoom=9,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    fig.show()

def plot_trajectory(traj, user_id=None, multiple=False):

    lat = np.array(traj["latitude"])
    lon = np.array(traj["longitude"])
    time = np.array(traj["time"])

    if multiple:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=lon,
                y=lat,
                mode="lines+markers",
                name=f"user {user_id}" if user_id else "trajet",
            )
        )

        return fig

    # ----------------------------------------------------------
    # SINGLE USER : MAP ANIMATION
    # ----------------------------------------------------------

    frames = []

    for i in range(len(lat)):

        frames.append(
            go.Frame(
                data=[

                    # trajectoire cumulée
                    go.Scattermapbox(
                        lat=lat[: i + 1],
                        lon=lon[: i + 1],
                        mode="lines",
                        line=dict(width=3, color="blue"),
                    ),

                    # point courant
                    go.Scattermapbox(
                        lat=[lat[i]],
                        lon=[lon[i]],
                        mode="markers",
                        marker=dict(size=12, color="red"),
                    ),
                ],
                name=str(i),
            )
        )

    # ----------------------------------------------------------
    # figure initiale
    # ----------------------------------------------------------

    fig = go.Figure(
        data=[

            go.Scattermapbox(
                lat=[lat[0]],
                lon=[lon[0]],
                mode="lines",
                line=dict(width=3, color="blue"),
                name="trajectory",
            ),

            go.Scattermapbox(
                lat=[lat[0]],
                lon=[lon[0]],
                mode="markers",
                marker=dict(size=12, color="red"),
                name="current",
            ),
        ],
        frames=frames,
    )

    # ----------------------------------------------------------
    # layout map
    # ----------------------------------------------------------
    
    lat_range = np.max(lat) - np.min(lat)
    lon_range = np.max(lon) - np.min(lon)

    max_range = max(lat_range, lon_range)

    zoom = 4
    
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=np.mean(lat) + 2,
                lon=np.mean(lon),
            ),
            zoom=zoom,
        ),
    )

    return fig

def plot_trajectory_medoid(traj, day=None, user_id=None):
    """
    traj: dict returned by build_trajectory
    """

    lat = np.array(traj["latitude"])
    lon = np.array(traj["longitude"])
    time = np.array(traj["time"])

    fig = go.Figure()

    # ----------------------------------------------------------
    # Trajectoire principale (ligne)
    # ----------------------------------------------------------

    fig.add_trace(
        go.Scattermapbox(
            lat=lat,
            lon=lon,
            mode="lines+markers",
            name="Trajectory",
            marker=dict(size=6),
            line=dict(width=2),
        )
    )

    # ----------------------------------------------------------
    # Medoid cell
    # ----------------------------------------------------------

    if traj.get("medoid_cell") is not None:

        med_lat, med_lon = traj["lat_medoid"], traj["lon_medoid"]

        fig.add_trace(
            go.Scattermapbox(
                lat=[med_lat],
                lon=[med_lon],
                mode="markers",
                name="Medoid",
                marker=dict(
                    size=14,
                    color="red",
                    symbol="circle",
                ),
            )
        )
        
        # ------------------------------------------------------
        # Cercle rayon de gyration
        # ------------------------------------------------------

        rg = traj["radius_of_gyration"]

        if rg is not None and rg > 0:

            circle_lats, circle_lons = utils.circle_coordinates(
                med_lat,
                med_lon,
                rg
            )

            fig.add_trace(
                go.Scattermapbox(
                    lat=circle_lats,
                    lon=circle_lons,
                    mode="lines",
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.15)",
                    line=dict(color="rgba(255,0,0,0.5)", width=2),
                    name="Radius of gyration",
                )
            )
            
    # ----------------------------------------------------------
    # Centre carte
    # ----------------------------------------------------------

    center_lat = np.mean(lat)
    center_lon = np.mean(lon)

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------

    title = "Trajectory"
    
    if day is not None:
        date = datetime.strptime(f"{day}/03/2014", "%d/%m/%Y")
        title += f" - {date.strftime('%A %d')}"

    if user_id is not None:
        title += f" | User {user_id}"

    if traj.get("radius_of_gyration") is not None:
        title += f" | RG = {traj['radius_of_gyration']:.1f} m"

    fig.update_layout(
        title=title,
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=center_lat,
                lon=center_lon,
            ),
            zoom=10,
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700,
    )

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
    traj_dict = utils.build_trajectory(events)
    fig_traj = plot_trajectory(traj_dict, user_id=user_id)
    # save_plotly_animation_gif(
    #     fig_traj,
    #     "gif/user_trajectory.gif"
    # )
    fig_dist = plot_distance(traj_dict, mode=mode)
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

        traj_dict = utils.build_trajectory(events_user)

        fig.add_trace(
            go.Scatter(
                x=traj_dict["latitude"],
                y=traj_dict["longitude"],
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

def save_plotly_animation_gif(
    fig,
    output_path="trajectory.gif",
    fps=2,
):
    """
    Sauvegarde une animation Plotly en GIF.
    """

    if fig.frames is None or len(fig.frames) == 0:
        raise ValueError("Figure has no animation frames")

    temp_dir = tempfile.mkdtemp()

    frame_paths = []

    # ----------------------------------------------------------
    # Export de chaque frame
    # ----------------------------------------------------------

    for i, frame in enumerate(fig.frames):

        frame_fig = go.Figure(
            data=frame.data,
            layout=fig.layout
        )

        frame_path = os.path.join(
            temp_dir,
            f"frame_{i:04d}.png"
        )

        pio.write_image(
            frame_fig,
            frame_path,
            width=900,
            height=700,
            scale=2
        )

        frame_paths.append(frame_path)

    # ----------------------------------------------------------
    # Création GIF
    # ----------------------------------------------------------

    images = [
        imageio.imread(p)
        for p in frame_paths
    ]

    imageio.mimsave(
        output_path,
        images,
        fps=fps
    )

    print(f"GIF saved to: {output_path}")
    
def plot_signal(signal, user, day, nb_records):    
    fig = px.line(
        x=np.arange(len(signal)) / 3600,
        y=signal,
        title=f"Number of records of the user {user} for each seconds of the day {day}/03/2014, total = {nb_records} records"
    )

    fig.update_layout(
        title_font=dict(size=24),
        xaxis_title="Time (hours)",
        yaxis_title="Number of records",
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
    )

    fig.update_xaxes(
        tickfont=dict(size=16),
        dtick=2,
        range=[0, 24]
    )

    fig.update_yaxes(
        tickfont=dict(size=16)
    )

    fig.show()

def plot_fft(freqs, power, harmonics, harmonic_power, user, day):
    fig = go.Figure()

    # FFT
    fig.add_trace(
        go.Scatter(
            x=freqs[1:],
            y=power[1:],
            mode="lines",
            name="FFT Power"
        )
    )

    # verticales rouges
    for f, p in zip(harmonics, harmonic_power):

        fig.add_vline(
            x=f,
            line_width=2,
            line_dash="dash",
            line_color="red"
        )

        # texte lisible
        fig.add_annotation(
            x=f,
            y=p,
            text=f"{f:.3f}",
            showarrow=True,
            arrowhead=1,
            yshift=15,
            font=dict(color="red")
        )

    fig.update_layout(
        title=f"FFT Spectrum User {user}; Day {day}",
        xaxis_title="Frequency",
        yaxis_title="Power",
        template="plotly_white",
        height=600
    )
    fig.show()

def plot_dupplicate_record_repartition(event, day):
    # Récupération du pic max de chaque utilisateur
    peaks = []

    for user in event:
        _, times = event[user]
        signal = utils.build_signal(times)
        peaks.append(np.max(signal))

    # Calcul des pics
    peaks = []

    for user in event:
        _, times = event[user]
        signal = utils.build_signal(times)
        peaks.append(np.max(signal))

    # Comptage exact
    counts = Counter(peaks)

    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    sns.set_theme()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=x, y=y)

    # Ajouter les valeurs au-dessus des barres
    for i, v in enumerate(y):
        ax.text(i, v + 0.5, str(v), ha='center')


    plt.xlabel("Max nb of records in 0s")
    plt.ylabel("Nb of users")
    plt.title(f"User distribution based on their max nb of records in 0s, nb of users {len(event)}, day {day}")
    
    plt.show()
    
def plot_mfcc_heatmap(mfcc, title="MFCC", figsize=(12, 4)):
    """
    Plot une heatmap des coefficients MFCC.

    Parameters
    ----------
    mfcc : np.ndarray
        Matrice de MFCC de forme (n_frames, n_mfcc),
        telle que retournée par extract_mfcc().
    title : str
        Titre du graphique.
    figsize : tuple
        Taille de la figure.
    """

    mfcc = np.asarray(mfcc)

    # Si le vecteur est 1D : (n_frames,) -> (n_frames, 1)
    if mfcc.ndim == 1:
        mfcc = mfcc.reshape(-1, 1)

    # mfcc est (frames, coefficients)
    # On transpose pour avoir (coefficients, frames)
    mfcc_plot = mfcc.T

    plt.figure(figsize=figsize)

    plt.imshow(
        mfcc_plot,
        aspect="auto",
        origin="upper",
        interpolation="nearest"
    )

    plt.colorbar(label="MFCC value")

    plt.xlabel("Frames")
    plt.ylabel("Coefficient MFCC")
    plt.title(title)

    # Afficher les indices des coefficients
    plt.yticks(np.arange(mfcc_plot.shape[0]))

    plt.tight_layout()
    plt.show()