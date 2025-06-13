# === this explorer can be operated at ===
# https://hentsch-phase-portraits.streamlit.app

#!/usr/bin/env python
# coding: utf-8

    # Run the Streamlit app
    # streamlit run phase_portraits_explorer_streamlit.py
    # Note: To run this script, use the command:
    # streamlit run phase_portraits_explorer_streamlit.py
    # in the terminal from the directory containing this file.
    # This will start a local server and open the app in your web browser.
    # The app will automatically update when you change the parameters in the sidebar.
    # Note: The app requires Matplotlib, NumPy, and Streamlit to be installed.
    # You can install them using pip:
    # pip install matplotlib numpy streamlit
    # Note: The app will not run in Jupyter Notebook or JupyterLab directly.
    # It is designed to be run as a standalone Streamlit app.
    # Note: The app will not save any figures to disk.
    # It will only display the figures in the Streamlit app.

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


import base64
import imageio.v2 as imageio
import tempfile
import os
# --- Phase portraits of the projected flow on the Hentsch screen ---
from pathlib import Path

import matplotlib.pyplot as plt  # ensure plt is matplotlib, not overwritten
import numpy as np
import streamlit as st
from matplotlib import cm, colors
from matplotlib.collections import LineCollection

st.set_page_config(layout="wide")

# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#  Constants
# ----------------------------------------------------------------------
SQRT6 = np.sqrt(6.0)
# ----------------------------------------------------------------------
#  Numerical safety thresholds (to prevent overflow/invalid warnings)
# ----------------------------------------------------------------------
OVERFLOW_LIMIT = 1e6      # clip trajectory coordinates to ± this value
RHO_SAT = 1e5             # cap radial distance used inside κ(ρ)
SPEED_SAT = 1e6           # cap speed magnitude for colour‑scaling
# FIGDIR = Path(__file__).with_suffix('').parent / 'phase_portraits'
# FIGDIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
#  Truncate a colormap to a specific range
# ----------------------------------------------------------------------
def truncate_cmap(
    cmap_name: str, min_val: float = 0.0, max_val: float = 1.0, n: int = 256
) -> colors.Colormap:
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
    parent_cmap = plt.get_cmap(cmap_name, n)  # use modern API; avoids deprecation
    new_colors = parent_cmap(np.linspace(min_val, max_val, n))
    return colors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_trunc_{min_val:.2f}_{max_val:.2f}", new_colors, N=n
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
    # Prevent κ(ρ) from becoming unreasonably large
    rho_clip = np.clip(rho, 0.0, RHO_SAT)
    kappa = rho_clip / SQRT6
    v1 = kappa * u1 + eps * np.cos(theta)
    v2 = kappa * u2 + eps * np.sin(theta)
    return v1, v2


# ----------------------------------------------------------------------
#  Phase-portrait plotting function
# ----------------------------------------------------------------------
def make_phase_plot(
    eps: float,
    theta: float,
    *,
    center_mode: str = "apex",
    show_lambda_circle: bool = True,
    show_null_circle: bool = True,
    show_eps_circle: bool = True,
    show_eig_line: bool = True,
    show_separatrix: bool = True,
    ngrid: int = 25,
    t_max: float = 200.0,
    nt: int = 300,
    seed_base: float = 2,
    seed_radius: float = 0.05,
    jitter: float = 0.1,
    view_span: float = 5.0,
    color_mode: str = "cosine",
    return_fig: bool = False,
    bidirectional: bool = False,
    t_min: float = 0.0,
):
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
    if color_mode not in {"cosine", "speed", "angle", "time", "accel", "curvature"}:
        raise ValueError(
            "color_mode must be 'cosine', 'speed', 'angle', 'time', 'accel', or 'curvature'"
        )

    # --------------------------------------------------------------
    #  Determine null point null_pt for the given ε, θ
    # --------------------------------------------------------------
    if abs(eps) < 1e-12:
        null_pt = np.array([0.0, 0.0])  # null point at origin
    else:
        rho_star = np.sqrt(abs(eps) * np.sqrt(6.0))  # ρ* = ε√6
        sign = -np.sign(eps)  # sign = -1 for ε > 0, +1 for ε < 0
        null_pt = np.array(
            [sign * rho_star * np.cos(theta), sign * rho_star * np.sin(theta)]
        )  # null point
    # --------------------------------------------------------------
    #  Set up the grid for the quiver plot
    # --------------------------------------------------------------
    half = view_span / 2.0
    # --------------------------------------------------------------
    #  Choose plot centre according to user selection
    #    "apex"      → (0,0)
    #    "null"      → field null point
    #    "midpoint"  → halfway between apex and null
    # --------------------------------------------------------------
    if center_mode == "apex":
        centre = np.array([0.0, 0.0])
    elif center_mode == "null":
        centre = null_pt.copy()
    else:  # "midpoint"
        centre = null_pt / 2.0
    x = np.linspace(centre[0] - half, centre[0] + half, ngrid)
    y = np.linspace(centre[1] - half, centre[1] + half, ngrid)
    X, Y = np.meshgrid(x, y)  # meshgrid for quiver plot
    U_raw, V_raw = vec_field(X, Y, eps, theta)  # raw vector field
    speed = np.hypot(U_raw, V_raw)
    speed = np.clip(speed, 0.0, SPEED_SAT)
    speed_max = speed.max() or 1.0  # avoid division by zero

    # normalise quiver vectors to unit length for uniform arrow size,
    # and flip direction for inward (cosα < 0) to indicate flow direction
    with np.errstate(invalid="ignore", divide="ignore"):
        # Compute cosine between u and v (still used for colour modes)
        dot = X * U_raw + Y * V_raw
        norm_u = np.hypot(X, Y)
        norm_v = np.hypot(U_raw, V_raw)
        cos_alpha = dot / (norm_u * norm_v + 1e-12)  # avoid divide by zero

        # --- keep the TRUE vector directions (no flipping) ---
        U = np.where(speed > 0, U_raw / speed, 0.0)
        V = np.where(speed > 0, V_raw / speed, 0.0)

    # --------------------------------------------------------------
    #  Set up the quiver plot
    # --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))  # create a figure and axis
    ax.set_aspect("equal")  # set aspect ratio to equal

    # --------------------------------------------------------------
    # Arrow length = (fraction_of_span) × view_span  (data units)
    # For unit‑normalised (U,V) and scale_units="xy",
    # displayed length  =  1 / scale_val  (in data units).
    # Choose fraction = 1/20 ⇒ arrow ≈ 5 % of view width.
    # --------------------------------------------------------------
    arrow_frac = 1.0 / 50.0  # 5% of view span
    arrow_len = arrow_frac * view_span
    scale_val = 1.0 / arrow_len  # quiver scale parameter
    # Keep shaft thickness a fixed fraction of the window width
    width_frac = arrow_frac * 0.15  # shaft ≈ 25 % of arrow length
    width_val = width_frac  # quiver 'width' is fraction of Axes width

    # quiver coloured by original speed magnitude, with fixed-size arrowheads
    q = ax.quiver(
        X,
        Y,
        U,
        V,
        speed,
        cmap="viridis",
        alpha=0.3,
        norm=colors.Normalize(vmin=0.0, vmax=speed_max),
        pivot="tail",
        angles="xy",
        scale_units="xy",
        scale=scale_val,
        width=width_val,  # <─ NEW: constant shaft width
        headwidth=3.0,  # points – constant on screen
        headlength=5.0,  # points – constant on screen
        headaxislength=3.5,  # points – constant on screen
    )

    # colour-bar to read magnitudes
    # cbar = fig.colorbar(q, ax=ax, shrink=0.8, pad=0.02, aspect=50)
    # cbar.set_label(r"$\|\widehat{\boldsymbol{V}}_{\epsilon,\theta}\|$")

    # set limits for the plot
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_xlabel("v₁")
    ax.set_ylabel("v₂")
    ax.set_title(
        f"ε = {eps:.3f},  θ = {theta/np.pi:.2f}π,  scalar = {color_mode}", fontsize=12
    )

    # ---------------------------------------------------------------
    #  Add visual aids to the plot
    # ---------------------------------------------------------------

    # eigendirection (stable / neutral) visual guide
    if show_eig_line:
        e_s = np.array([np.cos(theta), np.sin(theta)])  # stable eigendirection
        # length of the line is determined by the distance to the box edge
        if abs(e_s[0]) > 1e-12:
            t_x = half / abs(e_s[0])
        else:
            t_x = 0.0
        if abs(e_s[1]) > 1e-12:
            t_y = half / abs(e_s[1])
        else:
            t_y = 0.0
        # take the larger of the two distances so the line touches the box
        L = max(t_x, t_y)

        # choose the line centre: midpoint or null point
        line_centre = centre  # always null_pt

        ax.plot(
            [line_centre[0] - L * e_s[0], line_centre[0] + L * e_s[0]],
            [line_centre[1] - L * e_s[1], line_centre[1] + L * e_s[1]],
            lw=1.0,
            ls="-",
            color="black",
            alpha=0.8,
            zorder=3,
        )

        # --- improved label position for stable eigendirection annotation ---
        # Place label a fixed distance inside the plot boundary along the eigendirection
        margin_frac = 0.35  # place label just inside the view edge
        bound_x = centre[0] + half
        bound_y = centre[1] + half
        if abs(e_s[0]) > abs(e_s[1]):
            label_pos_x = bound_x - margin_frac
            label_pos_y = line_centre[1] + (label_pos_x - line_centre[0]) * e_s[1] / e_s[0]
        else:
            label_pos_y = bound_y - margin_frac
            label_pos_x = line_centre[0] + (label_pos_y - line_centre[1]) * e_s[0] / e_s[1]
        label_pos = np.array([label_pos_x, label_pos_y])
        ax.annotate(
            "stable\neigendirection",
            xy=(label_pos[0], label_pos[1]),
            xytext=(0, 0),
            textcoords="offset points",
            color="black",
            fontsize=8,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
        )
    # ------------------------------------------------------
    # visual λ-circle (ρ = 1.5)
    # if show_lambda_circle:
    #     ax.add_patch(
    #         plt.Circle(
    #             (0.0, 0.0), 1.5, ec="darkred", lw=1, ls=":", alpha=0.6, fill=False, zorder=2
    #         )
    #     )
    #     ax.annotate(
    #         "λ-circle  ρ = 3/2",
    #         xy=(0, 1),
    #         xytext=(0, 70),
    #         textcoords="offset points",
    #         ha="center",
    #         va="center",
    #         fontsize=12,
    #         color="darkred",
    #         weight="normal",
    #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
    #     )
    # dashed circle at the null radius ρ_null = sqrt(ε√6)
    rho_null = np.sqrt(abs(eps) * np.sqrt(6.0))
    if show_null_circle:
        ax.add_patch(
            plt.Circle(
                (0.0, 0.0),
                rho_null,
                ec="grey",
                lw=1,
                ls="--",
                alpha=0.6,
                fill=False,
                zorder=2,
            )
        )
        ax.annotate(
            "null circle",
            xy=(0, rho_null),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10,
            color="grey",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
        )
    # ------------------------------------------------------
    # visual circle ρ(null) = √6 ε
    # if show_eps_circle:
    #     ax.add_patch(
    #         plt.Circle(
    #             (0, 0),
    #             np.sqrt(6.0) * abs(eps),
    #             ec="purple",
    #             lw=1,
    #             ls=":",
    #             alpha=0.6,
    #             fill=False,
    #             zorder=2,
    #         )
    #     )
    #     ax.annotate(
    #         "balance circle",
    #         xy=(0, 0 + np.sqrt(6.0) * abs(eps)),
    #         xytext=(0, 14),
    #         textcoords="offset points",
    #         ha="center",
    #         va="center",
    #         fontsize=12,
    #         color="purple",
    #         weight="normal",
    #         bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
    #     )
    # ------------------------------------------------------
    # mark the origin and the field-null point
    ax.scatter(
        null_pt[0], null_pt[1], s=8, marker="o", c="white", edgecolors="k", zorder=4
    )
    ax.scatter(0.0, 0.0, s=8, marker="o", c="black", edgecolors="w", zorder=5)

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
    starts.extend(
        [
            (null_pt[0] + seed_radius * np.cos(a), null_pt[1] + seed_radius * np.sin(a))
            for a in angles
        ]
    )
    # -----------------------------------------------------------------------
    # (c) randomise the initial conditions
    if jitter > 0.0:
        jitter_abs = jitter * half
        noise = np.random.uniform(-jitter_abs, jitter_abs, size=(len(starts), 2))
        new_starts = []
        for (p_x, p_y), (n_x, n_y) in zip(starts, noise):
            rho_p = np.hypot(p_x, p_y)
            if abs(rho_p - 1.0) < 1e-6:  # λ-circle → no jitter
                new_starts.append((p_x, p_y))
            else:  # apply jitter
                new_starts.append(
                    (
                        np.clip(p_x + n_x, null_pt[0] - half, null_pt[0] + half),
                        np.clip(p_y + n_y, null_pt[1] - half, null_pt[1] + half),
                    )
                )
        starts = new_starts
    # -----------------------------------------------------------------------
    # --- the phase trajectories -----------------------------
    line_width = 0.5  # line width for the segments
    alpha_seg = 1  # alpha for the segments
    traj_store = []  # store the trajectories
    accel_max_traj = 0.0
    global_curv_values = []  # gather for global colour scale
    for u1_0, u2_0 in starts:
        if bidirectional:
            # integrate backward
            traj_backward = np.empty((nt, 2))
            u1, u2 = u1_0, u2_0
            for k in range(nt):
                traj_backward[k] = (u1, u2)
                v1, v2 = vec_field(u1, u2, eps, theta)
                u1 -= h * v1
                u2 -= h * v2
                u1 = np.clip(u1, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
                u2 = np.clip(u2, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
            # integrate forward
            traj_forward = np.empty((nt, 2))
            u1, u2 = u1_0, u2_0
            for k in range(nt):
                traj_forward[k] = (u1, u2)
                v1, v2 = vec_field(u1, u2, eps, theta)
                u1 += h * v1
                u2 += h * v2
                u1 = np.clip(u1, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
                u2 = np.clip(u2, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
            traj_combined = np.vstack((traj_backward[::-1][:-1], traj_forward))
            traj = traj_combined
            n_traj = traj.shape[0]
            scal = np.empty(n_traj - 1)
            scal_speed = np.empty(n_traj - 1)
            scal_angle = np.empty(n_traj - 1)
            scal_time = np.empty(n_traj - 1)
            scal_acc = np.empty(n_traj - 1)
            scal_curv = np.empty(n_traj - 1)
            sep_pts = []
            t = -t_max
            v1_prev, v2_prev = vec_field(traj[0, 0], traj[0, 1], eps, theta)
            for k in range(1, n_traj):
                u1, u2 = traj[k, 0], traj[k, 1]
                v1, v2 = vec_field(u1, u2, eps, theta)
                speed_seg = np.hypot(v1, v2) + 1e-12
                cos_alpha = (u1 * v1 + u2 * v2) / (np.hypot(u1, u2) * speed_seg)
                scal_speed[k - 1] = speed_seg
                scal[k - 1] = cos_alpha
                if k > 1:
                    if scal[k - 2] * cos_alpha < 0:
                        w = abs(scal[k - 2]) / (abs(scal[k - 2]) + abs(cos_alpha))
                        x_sep = (1 - w) * traj[k - 1, 0] + w * u1
                        y_sep = (1 - w) * traj[k - 1, 1] + w * u2
                        sep_pts.append((x_sep, y_sep))
                phi_seg = np.arctan2(u2, u1)
                scal_angle[k - 1] = phi_seg
                scal_time[k - 1] = t
                acc_seg = np.hypot(v1 - v1_prev, v2 - v2_prev) / h
                scal_acc[k - 1] = acc_seg
                accel_max_traj = max(accel_max_traj, acc_seg)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R11 = eps / np.sqrt(6.0) * (4 * cos_t**2 + sin_t**2)
                R22 = eps / np.sqrt(6.0) * (4 * sin_t**2 + cos_t**2)
                R12 = eps / np.sqrt(6.0) * 3 * cos_t * sin_t
                trace_R = 5 * eps / np.sqrt(6.0)
                shift = (1.0 / 6.0) - trace_R / 3.0
                S11, S22, S12 = R11 + shift, R22 + shift, R12
                r2 = u1 * u1 + u2 * u2 + 1e-12
                inv_r = 1.0 / np.sqrt(r2)
                nx, ny = u1 * inv_r, u2 * inv_r
                scal_curv[k - 1] = S11 * nx * nx + 2 * S12 * nx * ny + S22 * ny * ny
                v1_prev, v2_prev = v1, v2
                t += h
            global_curv_values.extend(scal_curv.tolist())
            segs = np.column_stack([traj[:-1], traj[1:]]).reshape(-1, 2, 2)
            traj_store.append(
                (
                    segs,
                    scal.copy(),
                    scal_speed.copy(),
                    scal_angle.copy(),
                    scal_time.copy(),
                    scal_acc.copy(),
                    scal_curv.copy(),
                    sep_pts,
                    alpha_seg,
                )
            )
        else:
            traj = np.empty((nt, 2))  # store the trajectory points
            scal = np.empty(nt - 1)  # directional cosine (cosine mode)
            scal_speed = np.empty(nt - 1)  # instantaneous speed   (speed mode)
            scal_angle = np.empty(nt - 1)  # polar angle  (angle mode)
            scal_time = np.empty(nt - 1)  # λ‑time         (time mode)
            scal_acc = np.empty(nt - 1)  # acceleration   (accel mode)
            scal_curv = np.empty(nt - 1)  # curvature scalar
            sep_pts = []  # store positions where ⟨u,ẋ⟩≈0
            u1, u2 = u1_0, u2_0
            traj[0] = (u1, u2)
            t = 0.0
            v1_prev, v2_prev = vec_field(u1, u2, eps, theta)
            for k in range(1, nt):  # time-stepping loop
                v1, v2 = vec_field(u1, u2, eps, theta)  # vector field
                u1 += h * v1
                u2 += h * v2
                u1 = np.clip(u1, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
                u2 = np.clip(u2, -OVERFLOW_LIMIT, OVERFLOW_LIMIT)
                t += h
                traj[k] = (u1, u2)
                speed_seg = np.hypot(v1, v2) + 1e-12
                cos_alpha = (u1 * v1 + u2 * v2) / (np.hypot(u1, u2) * speed_seg)
                scal_speed[k - 1] = speed_seg
                scal[k - 1] = cos_alpha
                if k > 1:
                    if scal[k - 2] * cos_alpha < 0:
                        w = abs(scal[k - 2]) / (abs(scal[k - 2]) + abs(cos_alpha))
                        x_sep = (1 - w) * traj[k - 1, 0] + w * u1
                        y_sep = (1 - w) * traj[k - 1, 1] + w * u2
                        sep_pts.append((x_sep, y_sep))
                phi_seg = np.arctan2(u2, u1)
                scal_angle[k - 1] = phi_seg
                scal_time[k - 1] = t
                acc_seg = np.hypot(v1 - v1_prev, v2 - v2_prev) / h
                scal_acc[k - 1] = acc_seg
                accel_max_traj = max(accel_max_traj, acc_seg)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R11 = eps / np.sqrt(6.0) * (4 * cos_t**2 + sin_t**2)
                R22 = eps / np.sqrt(6.0) * (4 * sin_t**2 + cos_t**2)
                R12 = eps / np.sqrt(6.0) * 3 * cos_t * sin_t
                trace_R = 5 * eps / np.sqrt(6.0)
                shift = (1.0 / 6.0) - trace_R / 3.0
                S11, S22, S12 = R11 + shift, R22 + shift, R12
                r2 = u1 * u1 + u2 * u2 + 1e-12
                inv_r = 1.0 / np.sqrt(r2)
                nx, ny = u1 * inv_r, u2 * inv_r
                scal_curv[k - 1] = S11 * nx * nx + 2 * S12 * nx * ny + S22 * ny * ny
                v1_prev, v2_prev = v1, v2
            global_curv_values.extend(scal_curv.tolist())
            segs = np.column_stack([traj[:-1], traj[1:]]).reshape(-1, 2, 2)
            traj_store.append(
                (
                    segs,
                    scal.copy(),
                    scal_speed.copy(),
                    scal_angle.copy(),
                    scal_time.copy(),
                    scal_acc.copy(),
                    scal_curv.copy(),
                    sep_pts,
                    alpha_seg,
                )
            )
    # --- color the segments by the directional cosines, speed, or angle
    norm_panel = colors.Normalize(vmin=-1.0, vmax=1.0)  # directional cosine ∈ [−1,1]
    accel_max = accel_max_traj or 1.0
    # scale factor so that acceleration colours span [0,1]
    accel_scale = 1
    if color_mode == "curvature":
        curv_arr = np.array(global_curv_values)
        if (curv_arr < 0).any():
            min_neg = np.percentile(curv_arr[curv_arr < 0], 5)
        else:
            min_neg = -1e-6
        if (curv_arr > 0).any():
            max_pos = np.percentile(curv_arr[curv_arr > 0], 95)
        else:
            max_pos = 1e-6
    for (
        segs,
        scal_cos,
        scal_spd,
        scal_ang,
        scal_tim,
        scal_acc,
        scal_curv,
        sep_pts,
        alpha_seg,
    ) in traj_store:
        if color_mode == "speed":
            scal_arr = scal_spd
            norm_seg = colors.Normalize(vmin=0.0, vmax=speed_max)
            cmap_this = "plasma"
        elif color_mode == "angle":
            scal_arr = scal_ang
            norm_seg = colors.Normalize(vmin=-np.pi, vmax=np.pi)
            cmap_this = "hsv"
        elif color_mode == "time":
            scal_arr = scal_tim
            norm_seg = colors.Normalize(vmin=0.0, vmax=t_max / 2)
            cmap_this = "inferno"
        elif color_mode == "accel":
            scal_arr = scal_acc * accel_scale  # map to [0,1]
            norm_seg = colors.NoNorm(vmin=0.0, vmax=1.0)
            cmap_this = "flag"
        elif color_mode == "curvature":
            scal_arr = scal_curv
            norm_seg = colors.TwoSlopeNorm(vmin=min_neg, vcenter=0.0, vmax=max_pos)
            cmap_this = "seismic"
        else:  # "cosine" (default)
            scal_arr = scal_cos
            norm_seg = norm_panel
            cmap_this = "jet"
        trunc_map = truncate_cmap(cmap_this, 0.0, 1.0)
        lc = LineCollection(segs, cmap=trunc_map, norm=norm_seg)
        lc.set_array(scal_arr)
        lc.set_linewidth(line_width)
        lc.set_alpha(alpha_seg)
        ax.add_collection(lc)

        # --- add the separatrix limaçon ---------------------------------
        # The separatrix (〈u,v〉 = 0) satisfies
        #       ρ^2 = -√6 ε cos(φ − θ)   with φ = arctan2(u₂,u₁)
        # This is real only when cos(φ−θ) ≤ 0, i.e. φ ∈ [θ+π/2, θ+3π/2].
        if show_separatrix and abs(eps) > 1e-12:
            phi_lim = np.linspace(
                theta + 0.5 * np.pi, theta + 1.5 * np.pi, 100, endpoint=True
            )
            rho_lim = np.sqrt(
                np.maximum(0.0, -np.sqrt(6.0) * eps * np.cos(phi_lim - theta))
            )
            x_lim = rho_lim * np.cos(phi_lim)
            y_lim = rho_lim * np.sin(phi_lim)
            ax.plot(x_lim, y_lim, lw=0.5, ls=":", color="darkgreen", alpha=0.6)
            ax.annotate(
                r"separatrix",
                xy=(x_lim[9], y_lim[9]),
                xytext=(x_lim[9], y_lim[9] + 0.2),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=12,
                color="darkgreen",
                weight="normal",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1),
            )

    # --------------------------------------------------------------
    #  Add legend (optional: for separatrix curve)
    # --------------------------------------------------------------
    # ax.legend(loc='upper right', fontsize=12)

    # --------------------------------------------------------------
    #  Plot the figure
    # --------------------------------------------------------------
    fig.tight_layout()
    if return_fig:
        return fig
    else:
        st.pyplot(fig)


# --- Streamlit UI in main() ---
def main():
    st.title("Screen Phase Portrait Explorer")
    st.sidebar.header("Phase Portrait Settings")
    color_mode = st.sidebar.selectbox(
        "**Diagnostic scalar**", ["cosine", "speed", "angle", "time", "accel", "curvature"],
        help="Choose how to color the trajectory segments.",
    )
    with st.sidebar.expander("**Parameters**", expanded=True):
            # -------- Define the interaction handlers for the number box and slider
        def box_changed():
            # fetch whichever key is present; default to 0.0
            val = st.session_state.get("eps_box", st.session_state.get("eps", 3/ (2 * np.sqrt(6))))
            st.session_state["eps"] = val
            st.session_state["eps_slider"] = val      # keep slider in sync
        def slider_changed():
            val = st.session_state.get("eps_slider", st.session_state.get("eps", 3/ (2 * np.sqrt(6))))
            st.session_state["eps"] = val
            st.session_state["eps_box"] = val         # keep number box in sync
        # ----- widgets -----
        val = st.session_state.get("eps", 3/ (2 * np.sqrt(6)))
        eps_min, eps_max = 0.0, 5.0
        eps_box = st.number_input(
            "Exact ε value",
            eps_min, eps_max,
            value=val,
            key="eps_box",
            step=0.005,
            format="%.3f",
            on_change=box_changed,
        )
        eps_slider = st.slider(
            "Shear Strength ε",
            eps_min, eps_max,
            value=val,
            key="eps_slider",
            step=0.005,
            format="%.3f",
            on_change=slider_changed,
        )
        eps = val

        # --- Synchronized theta controls in degrees and radians ---
        def theta_deg_box_changed():
            val = np.deg2rad(st.session_state.get("theta_deg_box", 0.0))
            st.session_state["theta_rad"] = val
        def theta_slider_changed():
            val = st.session_state.get("theta_slider_deg", 0.0)
            st.session_state["theta_deg_box"] = val
            st.session_state["theta_rad"] = np.deg2rad(val)

        # --- Ensure theta-related session state keys exist ---
        if "theta_rad" not in st.session_state:
            st.session_state["theta_rad"] = 0.0
            st.session_state["theta_deg_box"] = 0.0
            st.session_state["theta_slider_deg"] = 0.0

        theta_val_deg = np.rad2deg(st.session_state.get("theta_rad", 0.0))
        theta_deg_box = st.number_input(
            "Shear direction θ (degrees)",
            0.0, 360.0,
            value=theta_val_deg,
            key="theta_deg_box",
            step=1.0,
            on_change=theta_deg_box_changed,
        )
        theta_slider_deg = st.slider(
            "Shear direction θ (slider)",
            0.0, 360.0,
            value=theta_val_deg,
            key="theta_slider_deg",
            step=1.0,
            on_change=theta_slider_changed,
        )
        theta = st.session_state.get("theta_rad", 0.0)

    with st.sidebar.expander("**Visual aids**", expanded=False):
        show_separatrix    = st.checkbox("Separatrix", value=True)
        show_eig_line      = st.checkbox("Stable eigendirection line", value=False)
        show_null_circle   = st.checkbox("null circle (ρ = √(√6 ε)", value=False)
        # show_eps_circle    = st.checkbox("balance circle (ρ = √6 ε)", value=False)
        # show_lambda_circle = st.checkbox("λ‑circle (ρ = 3/2)", value=False)

    with st.sidebar.expander("**View Options**", expanded=True):
        ngrid = st.slider("vector field density", 10, 50, value=24, step=1)
        view_span = st.slider("View span (zoom)", 1.0, 10.0, value=6.0, step=0.1)
        center_mode = st.radio(
            "View centre",
            options=["apex", "null", "midpoint"],
            index=0,
            help="Choose which point the plot is centred on.",
        )

    with st.sidebar.expander("**Trajectory settings**", expanded=False):
        bidirectional = st.checkbox("Bidirectional trajectories (±λ)", value=True)
        if bidirectional:
            t_min = -st.slider("minimum negative time", 1.0, 10.0, value=1.0, step=1.0)
        else:
            t_min = 0.0
        t_max = st.slider(
            "maximum positive time", 1.0, 100.0, value=20.0, step=1.0
        )
        nt = st.slider(
            "time steps (trajectory resolution)", 50, 2000, value=300, step=50
        )
        seed_base = st.slider("trajectory seeds", 1, 720, value=180, step=1)
        seed_radius = st.slider(
            "seed distance from null point", 0.01, 1.0, value=0.5, step=0.005
        )
        jitter = st.slider("jitter (seeding noise)", 0.0, 0.3, value=0.0, step=0.01)

    make_phase_plot(
        eps,
        theta,
        ngrid=ngrid,
        t_max=t_max,
        t_min=t_min,
        nt=nt,
        seed_base=seed_base,
        seed_radius=seed_radius,
        jitter=jitter,
        view_span=view_span,
        color_mode=color_mode,
        center_mode=center_mode,
        # show_lambda_circle=show_lambda_circle,
        show_null_circle=show_null_circle,
        # show_eps_circle=show_eps_circle,
        show_eig_line=show_eig_line,
        show_separatrix=show_separatrix,
        bidirectional=bidirectional,
    )

    with st.sidebar.expander("**GIF Animation**", expanded=False):
        st.markdown("### Animation Parameters")

        animate_seed_base = st.checkbox("animate seed density", value=True)
        if animate_seed_base:
            seed_base_start = st.number_input("Seed base start", 1, 720, value=seed_base, step=1)
            seed_base_end = st.number_input("Seed base end", 1, 720, value=seed_base, step=1)
        else:
            seed_base_anim = seed_base # st.slider("Trajectory seeds", 1, 720, value=seed_base, step=1)

        animate_seed_radius = st.checkbox("animate seeding radius", value=True)
        if animate_seed_radius:
            seed_radius_start = st.number_input("Seed radius start", 0.01, 1.0, value=seed_radius, step=0.005, format="%.3f")
            seed_radius_end = st.number_input("Seed radius end", 0.01, 1.0, value=seed_radius, step=0.005, format="%.3f")
        else:
            seed_radius_anim = seed_radius # st.slider("Seed distance from null", 0.01, 0.4, value=seed_radius, step=0.005)
        eps_start = st.number_input("ε start", value=eps)
        eps_end = st.number_input("ε end", value=eps*2)
        theta_start_deg = st.number_input("θ start (degrees)", value=theta, step=1.0)
        theta_end_deg = st.number_input("θ end (degrees)", value=theta+90, step=1.0)
        theta_start = np.deg2rad(theta_start_deg)
        theta_end = np.deg2rad(theta_end_deg)
        n_frames = st.number_input("Number of increments", min_value=2, value=20, step=1)
        bounce = st.checkbox("Bounce (ping-pong loop)", value=True)

        generate = st.button("Generate Animated GIF")

        if generate:
            eps_vals = np.linspace(eps_start, eps_end, n_frames)
            theta_vals = np.linspace(theta_start, theta_end, n_frames)
            # Compute per-frame values for animated seeds and radius
            if animate_seed_base:
                seed_base_values = np.linspace(seed_base_start, seed_base_end, n_frames).astype(int)
            else:
                seed_base_values = [seed_base_anim] * n_frames

            if animate_seed_radius:
                seed_radius_values = np.linspace(seed_radius_start, seed_radius_end, n_frames)
            else:
                seed_radius_values = [seed_radius_anim] * n_frames

            output_file = generate_bounce_gif(
                eps_vals,
                theta_vals,
                ngrid=ngrid,
                t_max=t_max,
                t_min=t_min,
                nt=nt,
                seed_base_values=seed_base_values,
                seed_radius_values=seed_radius_values,
                jitter=jitter,
                view_span=view_span,
                color_mode=color_mode,
                center_mode=center_mode,
                show_null_circle=show_null_circle,
                show_eig_line=show_eig_line,
                show_separatrix=show_separatrix,
                bounce=bounce,
                bidirectional=bidirectional,
            )
            with open(output_file, "rb") as f:
                gif_bytes = f.read()
                b64_gif = base64.b64encode(gif_bytes).decode()

            st.download_button("Download Animated GIF", gif_bytes, file_name="phase_portrait_bounce.gif", mime="image/gif")
            st.markdown(
                f'<img src="data:image/gif;base64,{b64_gif}" style="width:100%;" loop autoplay>',
                unsafe_allow_html=True
            )


def generate_bounce_gif(
    eps_values,
    theta_values,
    filename="phase_portrait_bounce.gif",
    duration=0.1,
    bounce=True,
    seed_base_values=None,
    seed_radius_values=None,
    **kwargs
):
    with tempfile.TemporaryDirectory() as tmpdir:
        images = []
        progress = st.progress(0)
        status = st.empty()
        n_frames = len(eps_values)
        for i, (eps, theta) in enumerate(zip(eps_values, theta_values)):
            # Use per-frame seed_base and seed_radius if provided
            frame_kwargs = kwargs.copy()
            if seed_base_values is not None:
                frame_kwargs['seed_base'] = seed_base_values[i]
            if seed_radius_values is not None:
                frame_kwargs['seed_radius'] = seed_radius_values[i]
            fig = make_phase_plot(
                eps,
                theta,
                return_fig=True,
                **frame_kwargs,
            )
            path = os.path.join(tmpdir, f"frame_{len(images):03d}.png")
            fig.savefig(path)
            plt.close(fig)
            images.append(imageio.imread(path))
            progress.progress((i + 1) / n_frames)
            status.text(f"Generating frame {i + 1} of {n_frames}")
        if bounce:
            images = images + images[-2::-1]
        imageio.mimsave(filename, images, duration=duration, loop=0)
        progress.empty()
        status.empty()
        return filename


if __name__ == "__main__":
    main()
