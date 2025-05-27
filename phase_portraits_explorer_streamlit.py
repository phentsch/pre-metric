#!/usr/bin/env python
# coding: utf-8

# ## **Hentsch Screen Phase Portrait Explorer**
# ---
# This interactive tool allows you to explore the projected flow on the Hentsch screen for varying shear parameters and visualize key dynamic behaviors such as field nulls, eigendirections, and separatrices.
# ### **Controls**
# | Parameter        | Description |
# |------------------|-------------|
# | **ε (epsilon)**  | Shear strength controlling the deformation of the canonical flow. Critical value is \( \epsilon_c = 1/\sqrt{6} \). |
# | **θ (theta)**    | Shear direction angle, in radians. Direction of the applied shear deformation. |
# | **grid**         | Resolution of the background vector field (quiver) grid. Higher values show finer detail. |
# | **t_max**        | Maximum integration time for trajectories. Longer times reveal more global behavior. |
# | **nt**           | Number of time steps for trajectory integration. Affects smoothness of curves. |
# | **seeds**        | Number of trajectory seeds placed evenly around a circle near the field null. |
# | **seed r**       | Radius from the field null where seeds are initialized. Controls how close they start. |
# | **jitter**       | Amount of random noise added to seed positions. Helps reveal structural stability. |
# | **view**         | Width/height of the view window. Increasing this zooms out. |
# | **colour**       | How to colour trajectory curves:
#   - `cosine`: directional alignment of flow
#   - `speed`: instantaneous velocity
#   - `angle`: polar angle
#   - `time`: integration progress
#   - `accel`: instantaneous acceleration
# ---
# ### **Visual Aids**
# - **Black line**: stable (neutral) eigendirection
# - **Green dashed curve**: separatrix (trajectory boundary)
# - **Red dotted circle**: diagnostic λ-circle (\( \rho = 1 \))
# - **White and black dots**: field null point and origin
# ---
# ### **Tips**
# - Try increasing **ε** past \( \epsilon_c \approx 0.408 \) to see the emergence of a field null.
# - Adjust **θ** to rotate the shear direction and observe symmetry changes.
# - Add **jitter** to test robustness of the flow structure.
# - Use **colour mode = time** to reveal the speed at which trajectories evolve.

# In[1]:


# --- Phase portraits of the projected flow on the Hentsch screen ---
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
import streamlit as st
st.set_page_config(layout="wide")

# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------
SQRT6 = np.sqrt(6.0)
# FIGDIR = Path(__file__).with_suffix('').parent / 'phase_portraits'
# FIGDIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
#  Truncate a colormap to a specific range
# ----------------------------------------------------------------------
def truncate_cmap(cmap_name: str,
                  min_val: float = 0.0,
                  max_val: float = 1.0,
                  n: int = 256) -> colors.Colormap:
    """
    Parameters
    ----------
    cmap_name : str
        Name of the Matplotlib colormap.
    min_val, max_val : float
        End-points of the sub-range to keep (0 ≤ min_val < max_val ≤ 1).
    n : int
        Number of discrete colours in the truncated map.
    """
    if not (0.0 <= min_val < max_val <= 1.0):
        raise ValueError("min_val and max_val must satisfy 0 ≤ min < max ≤ 1")
    parent_cmap = cm.get_cmap(cmap_name, n)
    new_colors  = parent_cmap(np.linspace(min_val, max_val, n))
    return colors.LinearSegmentedColormap.from_list(
        f'{cmap_name}_trunc_{min_val:.2f}_{max_val:.2f}',
        new_colors,
        N=n
    )

# ----------------------------------------------------------------------
#  Vector field on the screen:  V̂_ε,θ(u) = κ(ρ) u + ε (cosθ, sinθ)
# ----------------------------------------------------------------------
def vec_field(u1: np.ndarray, u2: np.ndarray, eps: float, theta: float = 0.0):
    """
    Vector field on the Hentsch screen.
    V̂_ε,θ(u) = κ(ρ) u + ε (cosθ, sinθ)
    where κ(ρ) = ρ / √6 is the normalisation factor for the projected flow.
    Parameters
    ----------
    u1 : np.ndarray
        First component of the vector field.
    u2 : np.ndarray
        Second component of the vector field.
    eps : float
        Shear strength.
    theta : float
        Angle of the shear direction.
    Returns
    -------
    -------
    u1 : np.ndarray
        First component of the vector field.
    u2 : np.ndarray
        Second component of the vector field.
    """
    rho = np.hypot(u1, u2)
    kappa = rho / SQRT6                     # κ(ρ) = ρ / √6
    return (kappa * u1) + (eps * np.cos(theta)), (kappa * u2) + (eps * np.sin(theta))

# ----------------------------------------------------------------------
#  Phase-portrait plotting function
# ----------------------------------------------------------------------
def make_phase_plot(eps: float,
                    theta: float,
                    outfile: Path,
                    *,
                    ngrid: int = 25,
                    t_max: float = 200.0,
                    nt: int = 300,
                    seed_base: float = 2,
                    seed_radius: float = 0.05,
                    jitter: float = 0.1,
                    view_span: float = 5.0,
                    color_mode: str = "cosine",
                    save_fig: bool = False,
                    interactive: bool = True):
    """
    Generate a phase portrait of the projected flow on the Hentsch screen
    for a given value of ε and θ.
    Parameters
    ----------
    eps : float
        Shear strength.
    theta : float
        Angle of the shear direction.
    outfile : Path
        Output file path for the phase portrait.
    ngrid : int
        Number of grid points in each direction.
    t_max : float
        Maximum time for the simulation.
    nt : int
        Number of time steps.
    seed_base : int
        Base number of seeds for the initial conditions.
    seed_radius : float
        Radius of the seed ring for initial conditions.
    jitter : float
        Jitter for the initial conditions.
    view_span : float
        Span of the view in each direction.
    color_mode : {"cosine", "speed", "angle", "time", "accel"}
        How to colour trajectory segments:
        * ``"cosine"`` – by directional cosine 〈u, v〉/‖u‖‖v‖,
        * ``"speed"``  – by instantaneous speed ‖v‖,
        * ``"angle"``  – by polar angle φ = arctan2(u₂,u₁),
        * ``"time"``   – by integration time λ (0 → ``t_max``),
        * ``"accel"``  – by instantaneous acceleration magnitude ‖a‖.
    """
    if color_mode not in {"cosine", "speed", "angle", "time", "accel"}:
        raise ValueError("color_mode must be 'cosine', 'speed', 'angle', 'time', or 'accel'")

    # --------------------------------------------------------------
    #  Determine null point null_pt for the given ε, θ
    # --------------------------------------------------------------
    if abs(eps) < 1e-12:
        null_pt = np.array([0.0, 0.0]) # null point at origin
    else:
        rho_star = np.sqrt(abs(eps) * np.sqrt(6.0)) # ρ* = ε√6
        sign     = -np.sign(eps) # sign = -1 for ε > 0, +1 for ε < 0
        null_pt   = np.array([sign * rho_star * np.cos(theta),
                             sign * rho_star * np.sin(theta)]) # null point
    # --------------------------------------------------------------
    #  Set up the grid for the quiver plot
    # --------------------------------------------------------------
    half = view_span / 2.0
    # always center at null_pt (no center_mid param)
    centre = null_pt / 2.0
    x = np.linspace(centre[0] - half, centre[0] + half, ngrid)
    y = np.linspace(centre[1] - half, centre[1] + half, ngrid)
    X, Y = np.meshgrid(x, y) # meshgrid for quiver plot
    U_raw, V_raw = vec_field(X, Y, eps, theta) # raw vector field
    speed = np.hypot(U_raw, V_raw)
    speed_max = speed.max() or 1.0 # avoid division by zero

    # normalise quiver vectors to unit length for uniform arrow size,
    # and flip direction for inward (cosα < 0) to indicate flow direction
    with np.errstate(invalid='ignore', divide='ignore'):
        # Compute cosine between u and v at each grid point
        dot = X * U_raw + Y * V_raw
        norm_u = np.hypot(X, Y)
        norm_v = np.hypot(U_raw, V_raw)
        cos_alpha = dot / (norm_u * norm_v + 1e-12)  # avoid divide by zero

        # Flip direction for inward (cosα < 0)
        sign = np.where(cos_alpha < 0, -1.0, 1.0)

        # Normalise and apply directional sign
        U = np.where(speed > 0, sign * U_raw / speed, 0.0)
        V = np.where(speed > 0, sign * V_raw / speed, 0.0)
        # Option to keep the original speed magnitude
        # U = np.where(speed > 0, sign * U_raw, 0.0)
        # V = np.where(speed > 0, sign * V_raw, 0.0)

    # --------------------------------------------------------------
    #  Set up the quiver plot
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8,8)) # create a figure and axis
    ax.set_aspect('equal') # set aspect ratio to equal

    # use fixed scale for uniform arrow length
    scale_val = 10

    # quiver coloured by original speed magnitude
    q = ax.quiver(X, Y, U, V, speed,
                cmap='viridis',
                alpha=2/3,
                norm=colors.Normalize(vmin=0.0, vmax=speed_max),
                pivot='tail',
                angles='xy',
                scale_units='xy',
                scale=scale_val
                )

    # colour-bar to read magnitudes
    # cbar = fig.colorbar(q, ax=ax, shrink=0.8, pad=0.02, aspect=50)
    # cbar.set_label(r"$\|\widehat{\boldsymbol{V}}_{\epsilon,\theta}\|$")

    # set limits for the plot
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_xlabel(r"$v_1$")
    ax.set_ylabel(r"$v_2$")
    ax.set_title(rf"Hentsch Phase Portrait with $\quad \epsilon = {eps:.3f},\quad \theta = {theta/np.pi:.2f}\pi, \quad scalar: {color_mode} $")

    # ---------------------------------------------------------------
    #  Add visual aids to the plot
    # ---------------------------------------------------------------

    # eigendirection (stable / neutral) visual guide
    e_s = np.array([np.cos(theta), np.sin(theta)]) # stable eigendirection
    # length of the line is determined by the distance to the box edge
    if abs(e_s[0]) > 1e-12:
        t_x = half / abs(e_s[0])
    else:                           # direction is (0, ±1)
        t_x = 0.0
    if abs(e_s[1]) > 1e-12:
        t_y = half / abs(e_s[1])
    else:                           # direction is (±1, 0)
        t_y = 0.0
    # take the larger of the two distances so the line touches the box
    L = max(t_x, t_y)

    # choose the line centre: midpoint or null point
    line_centre = centre  # always null_pt

    ax.plot([line_centre[0] - L * e_s[0], line_centre[0] + L * e_s[0]],
            [line_centre[1] - L * e_s[1], line_centre[1] + L * e_s[1]],
            lw=1.5, ls='-', color='black', alpha=0.8, zorder=3)
    # annotate once per panel
    ax.annotate(
        "stable\neigendirection",
        xy=(line_centre[0] - 0.54 * L * e_s[0],
            line_centre[1] - 0.54 * L * e_s[1]),
        xytext=(0, 0), textcoords='offset points',
        color='black', fontsize=12, ha='center', va='center',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1)
    )
    # ------------------------------------------------------
    # visual λ-circle (ρ = 1)
    ax.add_patch(plt.Circle((0.0, 0.0), 1.0,
                            ec='darkred', lw=1, ls=':', alpha=0.6,
                            fill=False, zorder=2))
    ax.annotate(
        r"$\lambda$-circle  $(\rho = 1)$",
        xy=(0, 1),
        xytext=(0, 14),
        textcoords="offset points",
        ha="center", va="center", fontsize=12,
        color="darkred", weight="normal",
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1)
    )
    # ------------------------------------------------------
    # mark the origin and the field-null point
    ax.scatter(null_pt[0], null_pt[1], s=8, marker='o',
               c='white', edgecolors='k', zorder=4)
    ax.scatter(0.0, 0.0, s=8, marker='o',
               c='black', edgecolors='w', zorder=5)

    # ----------------------------------------------------------------
    #  Add phase trajectories
    # ----------------------------------------------------------------
    # --- sampling: single seed ring ---------------------------------
    h = t_max / nt  # time step
    starts = []
    # Use seed_base as number of seed points around the ring
    n_seeds = int(seed_base)
    angles = np.linspace(0.0, 2 * np.pi, n_seeds, endpoint=False)
    # Seed points placed at uniform angles around the null point
    starts.extend([
        (null_pt[0] + seed_radius * np.cos(a),
        null_pt[1] + seed_radius * np.sin(a)) for a in angles
    ])
    # -----------------------------------------------------------------------
    # (c) randomise the initial conditions
    if jitter > 0.0:
        jitter_abs = jitter * half
        noise = np.random.uniform(-jitter_abs, jitter_abs, size=(len(starts), 2))
        new_starts = []
        for (p_x, p_y), (n_x, n_y) in zip(starts, noise):
            rho_p = np.hypot(p_x, p_y)
            if abs(rho_p - 1.0) < 1e-6:            # λ-circle → no jitter
                new_starts.append((p_x, p_y))
            else:                                  # apply jitter
                new_starts.append((
                    np.clip(p_x + n_x, null_pt[0] - half, null_pt[0] + half),
                    np.clip(p_y + n_y, null_pt[1] - half, null_pt[1] + half)
                ))
        starts = new_starts
    # -----------------------------------------------------------------------
    # --- the phase trajectories -----------------------------
    line_width = 0.5 # line width for the segments
    alpha_seg = 1 # alpha for the segments
    traj_store = [] # store the trajectories
    accel_max_traj = 0.0
    for u1_0, u2_0 in starts:
        traj = np.empty((nt, 2)) # store the trajectory points
        scal = np.empty(nt-1)         # directional cosine (cosine mode)
        scal_speed = np.empty(nt-1)   # instantaneous speed   (speed mode)
        scal_angle = np.empty(nt-1)   # polar angle  (angle mode)
        scal_time  = np.empty(nt-1)   # λ‑time         (time mode)
        scal_acc   = np.empty(nt-1)   # acceleration   (accel mode)
        sep_pts = []                # store positions where ⟨u,ẋ⟩≈0
        u1, u2 = u1_0, u2_0
        traj[0] = (u1, u2)
        t = 0.0
        # ----- initial velocity for accel mode -----
        v1_prev, v2_prev = vec_field(u1, u2, eps, theta)
        for k in range(1, nt): # time-stepping loop
            v1, v2 = vec_field(u1, u2, eps, theta) # vector field
            u1 += h * v1
            u2 += h * v2
            t += h
            traj[k] = (u1, u2)
            # ----- directional cosine for colour -----
            speed_seg = np.hypot(v1, v2) + 1e-12 # avoid division by zero
            cos_alpha = (u1 * v1 + u2 * v2) / (np.hypot(u1, u2) * speed_seg)
            scal_speed[k-1] = speed_seg                 # store for "speed" mode
            scal[k-1] = cos_alpha                  # default cosine scalar
            # ---- detect separatrix crossing (cosα = 0) ----
            if k > 1:
                if scal[k-2] * cos_alpha < 0:        # sign change ⇒ crossed
                    # linear interpolation for better estimate
                    w = abs(scal[k-2]) / (abs(scal[k-2]) + abs(cos_alpha))
                    x_sep = (1 - w) * traj[k-1, 0] + w * u1
                    y_sep = (1 - w) * traj[k-1, 1] + w * u2
                    sep_pts.append((x_sep, y_sep))
            # ----- polar angle for "angle" mode -----
            phi_seg = np.arctan2(u2, u1)           # range (-π, π]
            scal_angle[k-1] = phi_seg
            # ----- λ-time for "time" mode -----
            scal_time[k-1] = t        # store current λ for "time" mode
            # ----- acceleration magnitude for "accel" mode -----
            acc_seg = np.hypot(v1 - v1_prev, v2 - v2_prev) / h
            scal_acc[k-1] = acc_seg
            accel_max_traj = max(accel_max_traj, acc_seg)
            v1_prev, v2_prev = v1, v2              # update for next step
        segs = np.column_stack([traj[:-1], traj[1:]]).reshape(-1, 2, 2)
        traj_store.append((segs,
                           scal.copy(),        # cosine
                           scal_speed.copy(),  # speed
                           scal_angle.copy(),  # angle
                           scal_time.copy(),   # time
                           scal_acc.copy(),    # accel
                           sep_pts,            # separatrix points
                           alpha_seg))
    # --- color the segments by the directional cosines, speed, or angle
    norm_panel = colors.Normalize(vmin=-1.0, vmax=1.0) # directional cosine ∈ [−1,1]
    accel_max = accel_max_traj or 1.0
    # scale factor so that acceleration colours span [0,1]
    accel_scale = 1
    for segs, scal_cos, scal_spd, scal_ang, scal_tim, scal_acc, sep_pts, alpha_seg in traj_store:
        if color_mode == "speed":
            scal_arr = scal_spd
            norm_seg = colors.Normalize(vmin=0.0, vmax=speed_max)
            cmap_this = 'plasma'
        elif color_mode == "angle":
            scal_arr = scal_ang
            norm_seg = colors.Normalize(vmin=-np.pi, vmax=np.pi)
            cmap_this = 'hsv'
        elif color_mode == "time":
            scal_arr = scal_tim
            norm_seg = colors.Normalize(vmin=0.0, vmax=t_max/2)
            cmap_this = 'inferno'
        elif color_mode == "accel":
            scal_arr = scal_acc * accel_scale     # map to [0,1]
            norm_seg = colors.NoNorm(vmin=0.0, vmax=1.0)
            cmap_this = 'flag'
        else:                         # "cosine" (default)
            scal_arr = scal_cos
            norm_seg = norm_panel
            cmap_this = 'jet'
        trunc_map = truncate_cmap(cmap_this, 0.02, 0.98)
        lc = LineCollection(segs,
                            cmap=trunc_map,
                            norm=norm_seg)
        lc.set_array(scal_arr)
        lc.set_linewidth(line_width)
        lc.set_alpha(alpha_seg)
        ax.add_collection(lc)


        # --- add the separatrix limaçon ---------------------------------
        # The separatrix (〈u,v〉 = 0) satisfies
        #       ρ^2 = -√6 ε cos(φ − θ)   with φ = arctan2(u₂,u₁)
        # This is real only when cos(φ−θ) ≤ 0, i.e. φ ∈ [θ+π/2, θ+3π/2].
        if abs(eps) > 1e-12:
            phi_lim = np.linspace(theta + 0.5*np.pi,
                                  theta + 1.5*np.pi,
                                  100, endpoint=True)
            rho_lim = np.sqrt(np.maximum(0.0,
                               -np.sqrt(6.0)*eps*np.cos(phi_lim - theta)))
            x_lim = rho_lim * np.cos(phi_lim)
            y_lim = rho_lim * np.sin(phi_lim)
            ax.plot(x_lim, y_lim,
                    lw=0.5, ls=':', color='darkgreen', alpha=0.6)
            ax.annotate(
                r"separatrix",
                xy=(x_lim[9], y_lim[9]),
                xytext=(x_lim[9], y_lim[9]+0.2),
                textcoords="data",
                ha="center", va="center", fontsize=12,
                color="darkgreen", weight="normal",
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1)
            )



    # --------------------------------------------------------------
    #  Add legend (optional: for separatrix curve)
    # --------------------------------------------------------------
    # ax.legend(loc='upper right', fontsize=12)

    # --------------------------------------------------------------
    #  Save the figure
    # --------------------------------------------------------------
    fig.tight_layout() # adjust layout to fit
    if interactive:
        import streamlit as st
        st.pyplot(fig)
    elif save_fig:
        fig.savefig(outfile, dpi=300) # save as PNG
        fig.savefig(outfile.with_suffix('.svg')) # save as SVG
        plt.close(fig) # close the figure to free memory


# ----------------------------------------------------------------------
#  Main driver - multiple regimes
# ----------------------------------------------------------------------
# def main():
#     color_mode = "cosine"   # "cosine", "speed", "angle", "time", "accel"
#     eps_c = 1.0 / SQRT6
#     eps_values = [
#         0.33*eps_c,
#         1.0*eps_c,
#         2.0*eps_c,
#         4.0*eps_c,
#         6.0*eps_c,
#     ]
#     theta = np.pi/3 # shear direction
#     for i, eps in enumerate(eps_values, 1):
#         fname = f"fig_phase_{color_mode}_{i}.png"
#         make_phase_plot(
#             eps, theta, FIGDIR / fname,
#             ngrid=25,
#             t_max=16,
#             nt=1000,
#             seed_base=5,
#             seed_radius=0.05,
#             jitter=0,
#             view_span=7.0,
#             color_mode=color_mode,
#             save_fig=False,
#             interactive=False
#         )


# --- Notebook widget interface ---

# --- Streamlit UI ---
eps = st.sidebar.slider("ε", 0.0, 4.0, value=0.4082, step=0.001)
theta = st.sidebar.slider("θ (rad)", 0.0, float(2*np.pi), value=0.0, step=0.01)
ngrid = st.sidebar.slider("Grid resolution", 10, 50, value=24, step=1)
t_max = st.sidebar.slider("t_max", 1.0, 100.0, value=20.0, step=1.0)
nt = st.sidebar.slider("nt", 50, 2000, value=300, step=50)
seed_base = st.sidebar.slider("Seeds", 1, 720, value=180, step=1)
seed_radius = st.sidebar.slider("Seed radius", 0.01, 0.2, value=0.05, step=0.005)
jitter = st.sidebar.slider("Jitter", 0.0, 0.3, value=0.0, step=0.01)
view_span = st.sidebar.slider("View span", 1.0, 10.0, value=5.0, step=0.1)
color_mode = st.sidebar.selectbox("Colour mode", ["cosine", "speed", "angle", "time", "accel"])

from io import BytesIO
fig_out = BytesIO()
import sys
import types
import __main__
import builtins
import os

import matplotlib.pyplot as plt
import matplotlib

imported_st = st

import matplotlib
import matplotlib.pyplot as plt

import sys
out = None
ui = None

if __name__ == "__main__":
    # Run the phase plot and display appropriately
    make_phase_plot(eps, theta, Path(), ngrid=ngrid, t_max=t_max, nt=nt,
                    seed_base=seed_base, seed_radius=seed_radius, jitter=jitter,
                    view_span=view_span, color_mode=color_mode,
                    save_fig=False, interactive=True)
    # Only display in notebook if not running in Streamlit
    if not getattr(st, "_is_running_with_streamlit", False):
        display(ui, out)
