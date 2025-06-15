# === this explorer can be operated at ===
# https://hentsch-pre-metric.streamlit.app

#!/usr/bin/env python
# coding: utf-8

# ## **Hentsch Manifold 3D Visual Explorer**
# ---
# This interactive 3D tool visualizes the structure of the Hentsch Manifold, including its algebraic null cone and canonical deformation fields. It is designed to support geometric intuition behind the projected and embedded flows described in the paper.
# ### **Controls**
# | Parameter        | Description |
# |------------------|-------------|
# | **ε (epsilon)**  | Shear strength controlling the linear deformation field. Critical shear occurs at \( \epsilon_c = 1/\sqrt{6} \). |
# | **θ (theta)**    | Shear direction angle (in parameter space). Rotates the shear field. |
# | **Q level**      | Level-set value \( Q(a, c, e) = q \) for deforming the cone surface. \( Q = 0 \) defines the null cone. |
# | **scale**        | Controls the overall spatial extent of the view. |
# | **Projection**   | Choose between perspective and orthographic projection. |
# | **Elev° / Azim°**| Controls the 3D camera elevation and azimuth angles. |
# | **show_cone**    | Displays the algebraic null cone surface (marching-cube extracted). |
# | **show_wireframe**| Displays a thin reference cone wireframe for orientation. |
# | **show_basis**   | Shows orthogonal shear basis vectors (in blue). |
# | **show_vectors3d**| Toggles display of the full 3D deformation vector field. |
# | **show_canonical_vector_field** | Toggle canonical (non-sheared) cone-preserving field. |
# | **show_shear_vector_field** | Toggle the shear-only vector field. |
# | **show_combined_vector_field** | Toggle the total (canonical + shear) vector field. |
# | **Axes a–f**     | Show/hide the individual coordinate axes in the \((a, c, e)\) and imaginary planes. |
# | **show_planes**  | Displays the three complex coordinate planes \((a+ib), (c+id), (e+if)\). |
# | **show_screen_planes** | Shows the ℓ⊥ screen planes used in projection. |
# ---
# ### **Visual Aids**
# - **ℓ axis** (color-coded): The degenerate axis of the null cone.
# - **Circular sections**: Represent constant-ℓ slices through the cone (diagnostic).
# - **Blue arrows**: Orthonormal basis vectors \(\vec{v}_1, \vec{v}_2\) in ℓ⊥ screen space.
# - **Colored quiver**: Visualizes vector field deformation.
# ---
# ### **Tips**
# - Toggle **combined vector field** and adjust **ε** to explore the onset of null point dynamics.
# - Use **θ** to rotate the deformation direction and test \( Z_3 \) symmetry.
# - Toggle **planes** and **axes** to understand the complex structure and how it aligns with real axes.
# - Set **Projection = ortho** to match analytic plots more closely.
# ---

# In[1]:


# --- Hentsch Manifold Explorer script ---
# ==============================================================================
# ==============================================================================
# IMPORTS
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
import matplotlib.colors as mcolors
import numpy.random as npr
from numpy.linalg import inv
import streamlit as st
import base64
import subprocess

st.set_page_config(layout="wide")
# ----------------------------------------------------------------------
# Safe initialisation of all Session State keys that callbacks rely on.
# These lines run on *every* script execution **before** any callbacks
# are triggered, so the keys are guaranteed to exist.
# ----------------------------------------------------------------------
for _k in ("quadratic_value", "q_box", "q_slider"):
    st.session_state.setdefault(_k, 0.0)

# Small value to avoid division by zero or overflow
EPS = 1e-12  # small value to avoid division by zero or overflow

# ==============================================================================
# CONFIGURATION (CFG)
# ==============================================================================
CFG = {
    # ---- figure / output --------------------------------------------
    "figsize": (6, 6),
    "dpi": 200,
    "save": False,
    "save_dpi": 300,
    "save_prefix": "figure_",

    # ---- plot bounds & axes -----------------------------------------
    "zoom": 1.0,
    "aspect_ratio": (1, 1, 1),

    # ---- arrow styling ----------------------------------------------
    "arrow_cone_height": 0.06,
    "arrow_cone_radius": 0.015,

    # ---- colours & cmaps --------------------------------------------
    "screen_cmap": "turbo",
    "field_cmap":  "viridis",
}

# ==============================================================================
# ==============================================================================
# DERIVED CONSTANTS
# ==============================================================================
figure_width, figure_height = CFG["figsize"]
figure_dpi                 = CFG["dpi"]
save_fig                   = CFG["save"]
save_dpi                   = CFG["save_dpi"]
color_map  = CFG["screen_cmap"]
field_cmap = CFG["field_cmap"]


# ==============================================================================
# HELPER CONSTANTS AND LABELS
# ==============================================================================
import os
from datetime import datetime
script_dir = os.path.dirname(os.getcwd())
span = np.linspace(-1, 1, 2)
tick_size = 5
plane_alpha = 0.1
ell_resolution = 250
cone_resolution = 25
cone_transparency = 0.3
reorient_vertical = False
field_canonical = 0
field_shear = 0
show_all = 1
show_real = 1
show_imag = 1
superposed = 0
show_acbf = 1
show_cebd = 1
show_eadf = 1
label_size = 6
x_label = 'a'
y_label = 'c'
z_label = 'e'
a_label = 'a'
b_label = 'b'
c_label = 'c'
d_label = 'd'
e_label = 'e'
f_label = 'f'
basis_vecs = [
    # Oriented to match the linear shear fields
    #   W₁ ≈ ( 0, -1,  1)
    #   W₂ ≈ (-1,  0,  1)
    #   W₃ ≈ ( 1, -1,  0)
    np.array([ 0, -1,  1], dtype=float),   # W₁ direction
    np.array([-1,  0,  1], dtype=float),   # W₂ direction
    np.array([ 1, -1,  0], dtype=float),   # W₃ direction
]
basis_vecs = [v / np.linalg.norm(v) for v in basis_vecs]   # normalise


# ==============================================================================
# GRAPHICAL CONVENTIONS
# ==============================================================================
# --- colors ---
col_real_x = '#FF0000'; col_real_y = '#00FF00'; col_real_z = '#0000FF'
col_imag_x = '#DDDD00'; col_imag_y = '#00FFFF'; col_imag_z = '#FF00FF'
lab_real_x = '#880000'; lab_real_y = '#008800'; lab_real_z = '#000088'
lab_imag_x = '#999900'; lab_imag_y = '#009999'; lab_imag_z = '#990099'
plane_ab = '#AAAA00'; plane_cd = '#00AAAA'; plane_ef = '#AA00AA'
# --- axes styling ---
line_width_real = 1.2; line_style_real = '-'
line_width_imag = 1.2; line_style_imag = ':'
# --- axis labels styling ---
font_size_real = 6; font_style_real = 'normal'; font_family_real = 'sans'
font_size_imag = 6; font_style_imag = 'italic'; font_family_imag = 'serif'
# ==============================================================================
# ROTATION MATRIX SETUP
# ==============================================================================
# --- Rotation matrix setup ---
ell_vec = np.array([1.0, 1.0, 1.0])
ell_unit = ell_vec / np.linalg.norm(ell_vec)
Rmat = np.eye(3)

# ==============================================================================
# PLOTTING FUNCTION
# ==============================================================================
import matplotlib.pyplot as plt

def make_plot(
    projection_type,
    elevation,
    azimuth,
    theta_deg,
    phi_deg,
    epsilon,
    quadratic_value,
    scale_factor,
    s_value,
    quiver_num,
    reorient_vertical,
    show_cone,
    show_wireframe,
    show_basis,
    show_vectors3d,
    show_canonical_vector_field,
    show_shear_vector_field,
    show_combined_vector_field,
    show_planes,
    show_screen_planes,
    show_axes_dict,
    quiver_normalize,
    uniform_quiver_seed,
    upper_nape_only,
    use_radial_coloring=False
):
    fig = plt.figure(figsize=CFG["figsize"], dpi=CFG["dpi"])
    ax = fig.add_subplot(111, projection='3d')
    global Rmat
    ax.cla()
    # Clamp epsilon to prevent extreme values
    epsilon = np.clip(epsilon, -1e3, 1e3)  # prevent extreme values
    theta_rad = np.deg2rad(theta_deg)
    phi_rad   = np.deg2rad(phi_deg)

    # Recompute geometry-related bounds and ticks using scale_factor
    plot_bounds = scale_factor * CFG["zoom"]
    x_bound_neg = y_bound_neg = z_bound_neg = -plot_bounds
    x_bound_pos = y_bound_pos = z_bound_pos = plot_bounds

    ticks = np.linspace(-scale_factor, scale_factor, 5)

    scale_axes = np.sqrt(3) if reorient_vertical else 1.0
    line_x_neg = x_bound_neg * scale_axes
    line_x_pos = x_bound_pos * scale_axes
    line_y_neg = y_bound_neg * scale_axes
    line_y_pos = y_bound_pos * scale_axes
    line_z_neg = z_bound_neg * scale_axes
    line_z_pos = z_bound_pos * scale_axes

    arrow_cone_height = CFG["arrow_cone_height"] * y_bound_pos
    arrow_cone_radius = CFG["arrow_cone_radius"] * x_bound_pos

    # Rotation matrix update
    ell_vec = np.array([1, 1, 1])
    ell_unit = ell_vec / np.linalg.norm(ell_vec)
    if reorient_vertical:
        target = np.array([0, 0, 1])
        v = np.cross(ell_unit, target)
        s = np.linalg.norm(v)
        c = ell_unit @ target
        if s < 1e-8:
            Rmat = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            Rmat = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    else:
        Rmat = np.eye(3)

    # --- Screen plane rotation/inverse ---
    def rotation_matrix_from_vectors(vec1, vec2):
        a, b = np.array(vec1)/np.linalg.norm(vec1), np.array(vec2)/np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        if s < 1e-10:
            return np.eye(3) if c > 0 else -np.eye(3)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

    R_align = rotation_matrix_from_vectors([1, 1, 1], [0, 0, 1])
    R_screen = np.eye(3) if reorient_vertical else inv(R_align)

    def _rot(p):
        p = np.asarray(p, dtype=float)
        if p.ndim == 1: return (Rmat @ p).ravel()
        if p.shape[-1] == 3: return p @ Rmat.T
        if p.shape[0] == 3: return Rmat @ p
        raise ValueError

    def _rot_screen(p):
        p = np.asarray(p, dtype=float)
        if p.ndim == 1:
            return (R_screen @ p).ravel()
        elif p.shape[-1] == 3:
            return p @ R_screen.T
        elif p.shape[0] == 3:
            return R_screen @ p
        else:
            raise ValueError("Invalid input to _rot_screen")

    # Labels + bounds
    x_label = r"$v_1$" if reorient_vertical else "a"
    y_label = r"$v_2$" if reorient_vertical else "c"
    z_label = r"$\ell$" if reorient_vertical else "e"
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.tick_params(axis='z', labelsize=tick_size)
    ax.set_xlim(-plot_bounds, plot_bounds)
    ax.set_ylim(-plot_bounds, plot_bounds)
    ax.set_zlim(-plot_bounds, plot_bounds)
    ax.set_box_aspect(CFG["aspect_ratio"])
    ax.set_proj_type(projection_type)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_title(
    f"θ = {theta_deg:.1f}°,  ϕ = {phi_deg:.1f}°,  ε = {epsilon:.3f},  Q = {quadratic_value:.2f},  scale = {scale_factor:.2f}",
    fontsize=6, loc='center'
)

    # --- Sync all toggle states from args ---
    show_a = show_axes_dict.get('a', True)
    show_b = show_axes_dict.get('b', True)
    show_c = show_axes_dict.get('c', True)
    show_d = show_axes_dict.get('d', True)
    show_e = show_axes_dict.get('e', True)
    show_f = show_axes_dict.get('f', True)
    show_acbf = show_cebd = show_eadf = show_planes

    # --- Axis Vectors ---
    def draw_cone(ax, base_center, direction, height=arrow_cone_height, radius=arrow_cone_radius, color='black'):
        resolution = 20
        theta = np.linspace(0, 2 * np.pi, resolution)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(theta)
        tip = np.array([0, 0, height])
        base = np.stack((x, y, z), axis=1)
        faces = [[tip, base[i], base[(i + 1) % resolution]] for i in range(resolution)]
        import numpy.linalg as la
        direction = np.array(direction, dtype=float)
        direction = direction / la.norm(direction)
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        if la.norm(v) < 1e-8:
            R = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (la.norm(v)**2))
        rotated_faces = [[[R @ point + np.asarray(base_center, dtype=float) for point in tri] for tri in faces]]
        for face_set in rotated_faces:
            ax.add_collection3d(Poly3DCollection(face_set, color=color))
    # --- Axis Vectors (a, b, c, d, e, f) ---
    if show_a:
        ax.plot(*_rot([[line_x_neg, line_x_pos], [0, 0], [0, 0]]), color=col_real_x, linestyle=line_style_real, linewidth=line_width_real)
        draw_cone(ax, base_center=_rot([line_x_pos, 0, 0]), direction=_rot([1, 0, 0]), color=col_real_x)
        ax.text(*_rot([1.1 * line_x_pos, 0, 0]), a_label, fontsize=font_size_real, color=lab_real_x, fontstyle=font_style_real, fontfamily=font_family_real)
    if show_c:
        ax.plot(*_rot([[0, 0], [line_y_neg, line_y_pos], [0, 0]]), color=col_real_y, linestyle=line_style_real, linewidth=line_width_real)
        draw_cone(ax, base_center=_rot([0, line_y_pos, 0]), direction=_rot([0, 1, 0]), color=col_real_y)
        ax.text(*_rot([0, 1.1 * line_y_pos, 0]), c_label, fontsize=font_size_real, color=lab_real_y, fontstyle=font_style_real, fontfamily=font_family_real)
    if show_e:
        ax.plot(*_rot([[0, 0], [0, 0], [line_z_neg, line_z_pos]]), color=col_real_z, linestyle=line_style_real, linewidth=line_width_real)
        draw_cone(ax, base_center=_rot([0, 0, line_z_pos]), direction=_rot([0, 0, 1]), color=col_real_z)
        ax.text(*_rot([0, 0, 1.1 * line_z_pos]), e_label, fontsize=font_size_real, color=lab_real_z, fontstyle=font_style_real, fontfamily=font_family_real)
    if show_b:
        if superposed:
            x_b, y_b, z_b = [0, 0], [line_y_neg, line_y_pos], [0, 0]
            base_b, dir_b = [0, line_y_pos, 0], [0, 1, 0]
            text_b = (0, 1.2 * line_y_pos, 0)
        else:
            x_b, y_b, z_b = [line_x_pos, line_x_pos], [0, line_y_pos], [0, 0]
            base_b, dir_b = [line_x_pos, line_y_pos, 0], [0, 1, 0]
            text_b = (1 * line_x_pos, 1.1 * line_y_pos, 0)
        ax.plot(*_rot([x_b, y_b, z_b]), color=col_imag_x, linestyle=line_style_imag, linewidth=line_width_imag)
        draw_cone(ax, base_center=_rot(base_b), direction=_rot(dir_b), color=col_imag_x)
        ax.text(*_rot(text_b), b_label, fontsize=font_size_imag, color=lab_imag_x, fontstyle=font_style_imag, fontfamily=font_family_imag)
    if show_d:
        if superposed:
            x_d, y_d, z_d = [0, 0], [0, 0], [line_z_neg, line_z_pos]
            base_d, dir_d = [0, 0, line_z_pos], [0, 0, 1]
            text_d = (0, 0, 1.2 * line_z_pos)
        else:
            x_d, y_d, z_d = [0, 0], [line_y_pos, line_y_pos], [0, line_z_pos]
            base_d, dir_d = [0, line_y_pos, line_z_pos], [0, 0, 1]
            text_d = (0, line_y_pos, 1.1 * line_z_pos)
        ax.plot(*_rot([x_d, y_d, z_d]), color=col_imag_y, linestyle=line_style_imag, linewidth=line_width_imag)
        draw_cone(ax, base_center=_rot(base_d), direction=_rot(dir_d), color=col_imag_y)
        ax.text(*_rot(text_d), d_label, fontsize=font_size_imag, color=lab_imag_y, fontstyle=font_style_imag, fontfamily=font_family_imag)
    if show_f:
        if superposed:
            x_f, y_f, z_f = [line_x_neg, line_x_pos], [0, 0], [0, 0]
            base_f, dir_f = [line_x_pos, 0, 0], [1, 0, 0]
            text_f = (1.2 * line_x_pos, 0, 0)
        else:
            x_f, y_f, z_f = [0, line_x_pos], [0, 0], [line_z_pos, line_z_pos]
            base_f, dir_f = [line_x_pos, 0, line_z_pos], [1, 0, 0]
            text_f = (1.1 * line_x_pos, 0, line_z_pos)
        ax.plot(*_rot([x_f, y_f, z_f]), color=col_imag_z, linestyle=line_style_imag, linewidth=line_width_imag)
        draw_cone(ax, base_center=_rot(base_f), direction=_rot(dir_f), color=col_imag_z)
        ax.text(*_rot(text_f), f_label, fontsize=font_size_imag, color=lab_imag_z, fontstyle=font_style_imag, fontfamily=font_family_imag)

    # --- Basis Vectors ---
    if show_basis:
        for bv in basis_vecs:
            tip_blue = _rot(bv * 0.75)
            ax.quiver(0, 0, 0,
                      tip_blue[0], tip_blue[1], tip_blue[2],
                      color='darkgreen', linewidth=1.2, arrow_length_ratio=0.08,
                      alpha=0.9)

    # --- Complex Planes ---
    if show_planes:
        X1, Y1 = np.meshgrid(span, span)
        Z1 = np.zeros_like(X1)
        X1r, Y1r, Z1r = _rot(np.array([X1.flatten(), Y1.flatten(), Z1.flatten()]).T).T.reshape(3, *X1.shape)
        ax.plot_surface(X1r, Y1r, Z1r, alpha=plane_alpha, color=plane_ab)
    if show_planes:
        Y2, Z2 = np.meshgrid(span, span)
        X2 = np.zeros_like(Y2)
        X2r, Y2r, Z2r = _rot(np.array([X2.flatten(), Y2.flatten(), Z2.flatten()]).T).T.reshape(3, *Y2.shape)
        ax.plot_surface(X2r, Y2r, Z2r, alpha=plane_alpha, color=plane_cd)
    if show_planes:
        Z3, X3 = np.meshgrid(span, span)
        Y3 = np.zeros_like(Z3)
        X3r, Y3r, Z3r = _rot(np.array([X3.flatten(), Y3.flatten(), Z3.flatten()]).T).T.reshape(3, *Z3.shape)
        ax.plot_surface(X3r, Y3r, Z3r, alpha=plane_alpha, color=plane_ef)

    # --- Core Geometry: ℓ Line and Cone ---
    # Central (1,1,1) axis line
    ell = np.array([1, 1, 1])
    if reorient_vertical:
        ell = ell / np.linalg.norm(ell)
    ell_vals = np.linspace(x_bound_neg, z_bound_pos, ell_resolution)
    line_pts = np.outer(ell_vals, ell)
    ell_a, ell_c, ell_e = line_pts[:, 0], line_pts[:, 1], line_pts[:, 2]
    colour_raw_line = ell_a + ell_c + ell_e
    color_min = colour_raw_line.min()
    color_max = colour_raw_line.max()
    colour_norm_line = (colour_raw_line - color_min) / (color_max - color_min)
    ell_rot = _rot(line_pts)
    ax.scatter(ell_rot[:, 0], ell_rot[:, 1], ell_rot[:, 2], c=colour_norm_line, cmap=color_map, s=1.5, alpha=1)
    cmap = plt.get_cmap(color_map)
    arrow_colour = cmap((colour_raw_line[-1] - color_min) / (color_max - color_min))
    draw_cone(ax, base_center=_rot(line_pts[-1]), direction=_rot(ell), height=arrow_cone_height, radius=arrow_cone_radius, color=arrow_colour)
    ax.text(*_rot(line_pts[-1] + 0.1 * x_bound_pos), r'$\ell$', fontsize=8, color=arrow_colour)

    # --- Optional Features: Wireframe, Screen Planes, Vector Fields ---
    # Wireframe reference cone
    if show_wireframe:
        wireframe_scale = 1 if reorient_vertical else np.sqrt(3)/2
        wire_r = np.linspace(0, z_bound_pos, 11)
        wire_theta = np.linspace(0, 2 * np.pi, 73)
        Rg, Tg = np.meshgrid(wire_r, wire_theta, indexing='ij')
        Rg = Rg * wireframe_scale
        Xw = Rg * np.sqrt(2) * np.cos(Tg)
        Yw = Rg * np.sqrt(2) * np.sin(Tg)
        Zw = Rg
        z_axis = np.array([0.0, 0.0, 1.0])
        target = ell_unit
        v = np.cross(z_axis, target)
        s = np.linalg.norm(v)
        c = z_axis @ target
        if s < 1e-8:
            R_align = np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
        pts_upper = np.column_stack([Xw.flatten(), Yw.flatten(), Zw.flatten()])
        pts_lower = np.column_stack([Xw.flatten(), Yw.flatten(), -Zw.flatten()])
        if reorient_vertical:
            R_final = R_align @ Rmat
        else:
            R_final = R_align
        pts_upper_r = (pts_upper @ R_final.T).reshape(*Xw.shape, 3)
        pts_lower_r = (pts_lower @ R_final.T).reshape(*Xw.shape, 3)
        Xu, Yu, Zu = pts_upper_r[..., 0], pts_upper_r[..., 1], pts_upper_r[..., 2]
        Xl, Yl, Zl = pts_lower_r[..., 0], pts_lower_r[..., 1], pts_lower_r[..., 2]
        ax.plot_wireframe(Xu, Yu, Zu, color='black', linewidth=0.1, alpha=1)
        ax.plot_wireframe(Xl, Yl, Zl, color='black', linewidth=0.1, alpha=1)

    # Quadratic null cone
    if show_cone:
        if reorient_vertical:
            x = np.linspace(x_bound_neg, x_bound_pos, cone_resolution)
            y = np.linspace(y_bound_neg, y_bound_pos, cone_resolution)
            z = np.linspace(z_bound_neg, z_bound_pos, cone_resolution)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            v = np.array([1, 1, 1])
            v = v / np.linalg.norm(v)
            z_axis = np.array([0, 0, 1])
            def rotation_matrix_from_vectors(vec1, vec2):
                a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
                v = np.cross(a, b)
                c = np.dot(a, b)
                s = np.linalg.norm(v)
                if s < 1e-10:
                    return np.eye(3) if c > 0 else -np.eye(3)
                kmat = np.array([[0, -v[2], v[1]],
                                 [v[2], 0, -v[0]],
                                 [-v[1], v[0], 0]])
                return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
            R = rotation_matrix_from_vectors(v, z_axis)
            R_inv = np.linalg.inv(R)
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            Z_flat = Z.flatten()
            coords = np.stack([X_flat, Y_flat, Z_flat], axis=1)
            rotated_coords = coords @ R_inv.T
            Xr, Yr, Zr = rotated_coords[:, 0].reshape(X.shape), rotated_coords[:, 1].reshape(Y.shape), rotated_coords[:, 2].reshape(Z.shape)
            F = Xr * Yr + Yr * Zr + Zr * Xr
        else:
            x = np.linspace(x_bound_neg, x_bound_pos, cone_resolution)
            y = np.linspace(y_bound_neg, y_bound_pos, cone_resolution)
            z = np.linspace(z_bound_neg, z_bound_pos, cone_resolution)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            F = X * Y + Y * Z + Z * X
        verts, faces, normals, values = marching_cubes(
            F, level=quadratic_value,
            spacing=(x[1] - x[0], y[1] - y[0], z[1] - z[0])
        )
        verts += np.array([x[0], y[0], z[0]])
        if reorient_vertical:
            proj = verts[:, 2]
        else:
            proj = verts @ ell_unit
        proj_normalized = (proj - proj.min()) / (proj.max() - proj.min())
        mesh = Poly3DCollection(verts[faces], alpha=cone_transparency, edgecolor='k', linewidths=0.05)
        mesh.set_array(proj_normalized[faces].mean(axis=1))
        mesh.set_cmap(color_map)
        ax.add_collection3d(mesh)

    # Screen Planes and Cross-sections
    if show_screen_planes:
        cmap = plt.get_cmap(color_map)
        norm = plt.Normalize(vmin=-1, vmax=1)
        screen_values = [s_value]
        for ell_value in screen_values:
            ell_radius = np.sqrt(2) * ell_value
            screen_range = np.linspace(-1, 1, 2)
            X_screen, Y_screen = np.meshgrid(screen_range, screen_range)
            Z_screen = np.full_like(X_screen, ell_value)
            coords = np.stack([X_screen.flatten(), Y_screen.flatten(), Z_screen.flatten()], axis=1)
            rotated = _rot_screen(coords)
            X_screen = rotated[:, 0].reshape(X_screen.shape)
            Y_screen = rotated[:, 1].reshape(Y_screen.shape)
            Z_screen = rotated[:, 2].reshape(Z_screen.shape)
            screen_color = cmap(norm(ell_value))
            ax.plot_surface(X_screen, Y_screen, Z_screen, alpha=0.2, color=screen_color, edgecolor='none')

            annot_pos = _rot_screen(np.array([-0.96, -0.96, ell_value]))
            ax.text(*annot_pos, r'$\ell^\perp$', fontsize=10, color=screen_color)

            theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = ell_radius * np.cos(theta)
            y_circle = ell_radius * np.sin(theta)
            z_circle = np.full_like(x_circle, ell_value)
            circ_pts = _rot_screen(np.stack([x_circle, y_circle, z_circle], axis=1))
            for i in range(len(theta) - 1):
                ax.plot(circ_pts[i:i+2, 0], circ_pts[i:i+2, 1], circ_pts[i:i+2, 2],
                        color=cmap(norm(0.5 * (circ_pts[i, 2] + circ_pts[i+1, 2]))), linewidth=1.2)

            label_color = cmap(norm(ell_value))
            v1_arrow_start = _rot_screen(np.array([0, 0, ell_value]))
            v1_arrow_dir = _rot_screen(np.array([0, 1, 0]))
            v1_arrow_tip = v1_arrow_start + 1.1 * v1_arrow_dir
            v2_arrow_dir = _rot_screen(np.array([-1.1, 0, 0]))
            v2_arrow_tip = v1_arrow_start + v2_arrow_dir
            ax.quiver(*v1_arrow_start, *v1_arrow_dir, color=label_color, arrow_length_ratio=0.1, linewidth=1)
            ax.text(*v1_arrow_tip, r'$\vec{v}_2$', fontsize=8, color=label_color)
            ax.quiver(*v1_arrow_start, *v2_arrow_dir, color=label_color, arrow_length_ratio=0.1, linewidth=1)
            ax.text(*v2_arrow_tip, r'$\vec{v}_1$', fontsize=8, color=label_color)

            radius_arrow_start = _rot_screen(np.array([0, 0, ell_value]))
            radius_arrow_dir = _rot_screen(np.array([0, ell_radius, 0]))
            radius_label_pos = _rot_screen(np.array([0, 0.25 * ell_radius, ell_value + 0.075]))
            ax.quiver(*radius_arrow_start, *radius_arrow_dir, color='k', arrow_length_ratio=0, linewidth=0.6, ls=':')
            ax.text(*radius_label_pos, r'$r = \sqrt{2}\, \ell$', fontsize=8, color='k')

            hyp_start = _rot_screen(np.array([0, 0, 0]))
            hyp_dir = _rot_screen(np.array([0, ell_radius, ell_value]))
            adj_dir = _rot_screen(np.array([0, 0, ell_value]))
            angle_label_pos = _rot_screen(np.array([0, 0.075, 0.15 * ell_value]))
            ax.quiver(*hyp_start, *hyp_dir, color='k', arrow_length_ratio=0, linewidth=0.6, ls=':')
            ax.quiver(*hyp_start, *adj_dir, color='k', arrow_length_ratio=0, linewidth=3)
            ax.text(*angle_label_pos, r'$\theta = \tan^{-1}(\sqrt{2})$', fontsize=8, color='k')

    # 3D cone-preserving vector field
    s_max = max(x_bound_pos, y_bound_pos, z_bound_pos) if reorient_vertical else np.sqrt(3)/2 * max(x_bound_pos, y_bound_pos, z_bound_pos)
    v1 = np.array([1, -1, 0]) / np.sqrt(2)
    v2 = np.array([1, 1, -2]) / np.sqrt(6)

    if uniform_quiver_seed:
        N_slices = max(4, int(quiver_num / 36))
        N_rings = max(4, int(quiver_num / 36))
        pts = []
        s_vals = np.linspace(-s_max, s_max, N_slices)
        s_vals = s_vals[np.abs(s_vals) > 1e-4]  # avoid s ≈ 0 which gives r ≈ 0
        for s in s_vals:
            r = np.sqrt(2) * abs(s)
            num_theta = max(4, int(N_rings * r / s_max))
            theta_vals = np.linspace(0, 2*np.pi, num_theta, endpoint=True)
            for theta in theta_vals:
                p = (
                    s * ell_unit +
                    r * np.cos(theta) * v1 +
                    r * np.sin(theta) * v2
                )
                pts.append(p)
        pts = np.array(pts)
    else:
        rng = npr.default_rng(8)
        N_pts = quiver_num
        radial_bias = 0.5
        u_rand = rng.random(N_pts)
        r_vals = s_max * u_rand**radial_bias
        signs = rng.choice([-1, 1], size=N_pts)
        s_vals = signs * r_vals
        theta_vals = rng.uniform(0, 2*np.pi, N_pts)
        alpha_vals = np.sqrt(2) * s_vals * np.cos(theta_vals)
        beta_vals  = np.sqrt(2) * s_vals * np.sin(theta_vals)
        pts = (
            s_vals[:,None]  * ell_unit +
            alpha_vals[:,None] * v1 +
            beta_vals[:,None]  * v2
        )

    # --- Vector field construction (vecs_list population) ---
    vecs_list = []
    if show_canonical_vector_field:
        def V_can(a, c, e):
            return np.array([a * a - c * e, c * c - a * e, e * e - a * c])
        vecs_list.append(np.array([V_can(*p) for p in pts]))
    if show_shear_vector_field:
        def W1(a, c, e):
            return np.array([0, -a - c, a + e])
        def W3(a, c, e):
            return np.array([c - e, e - a, a - c])
        def W2(a, c, e):
            return np.array([-a - c, 0, c + e])
        def W_theta(a, c, e, th=theta_rad):
            alpha  = np.sin(phi_rad)
            beta   = np.cos(phi_rad) * np.sin(th)
            gamma_ = np.cos(phi_rad) * np.cos(th)
            return (alpha  * W1(a, c, e) +
                    beta   * W2(a, c, e) +
                    gamma_ * W3(a, c, e))
        vecs_list.append(np.array([W_theta(*p) for p in pts]))
    if show_combined_vector_field:
        def W1(a, c, e):
            return np.array([0, -a - c, a + e])
        def V_can(a, c, e):
            return np.array([a * a - c * e, c * c - a * e, e * e - a * c])
        def W3(a, c, e):
            return np.array([c - e, e - a, a - c])
        def W2(a, c, e):
            return np.array([-a - c, 0, c + e])
        def W_theta(a, c, e, th=theta_rad):
            alpha  = np.sin(phi_rad)
            beta   = np.cos(phi_rad) * np.sin(th)
            gamma_ = np.cos(phi_rad) * np.cos(th)
            return (alpha  * W1(a, c, e) +
                    beta   * W2(a, c, e) +
                    gamma_ * W3(a, c, e))
        def V_combined(a, c, e, eps=epsilon, th=theta_rad):
            return V_can(a, c, e) + eps * W_theta(a, c, e, th)
        vecs_list.append(np.array([V_combined(*p) for p in pts]))

    if upper_nape_only:
        mask = (pts[:, 0] + pts[:, 1] + pts[:, 2]) >= 0  # filter points with a + c + e >= 0
        pts = pts[mask]
        vecs_list = [v[mask] for v in vecs_list]
    for vecs in vecs_list:
        pts_rot = _rot(pts)
        vecs_rot = _rot(pts + vecs) - pts_rot
        mag = np.linalg.norm(vecs_rot, axis=1)
        mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
        vecs_rot = np.nan_to_num(vecs_rot, nan=0.0, posinf=0.0, neginf=0.0)

        if use_radial_coloring:
            dot = np.sum(pts * vecs, axis=1)
            norm_pos = np.linalg.norm(pts, axis=1)
            norm_vec = np.linalg.norm(vecs, axis=1)
            denom = np.where(norm_pos * norm_vec < 1e-12, 1e-12, norm_pos * norm_vec)
            cosine_scalar = dot / denom
            norm_metric = mcolors.Normalize(vmin=-1, vmax=1)
            cmap = 'jet'
            array = cosine_scalar
        else:
            norm_metric = mcolors.Normalize(vmin=mag.min(), vmax=mag.max() * 1.2)
            cmap = field_cmap
            array = mag

        ax.quiver(
            pts_rot[:, 0], pts_rot[:, 1], pts_rot[:, 2],
            vecs_rot[:, 0], vecs_rot[:, 1], vecs_rot[:, 2],
            length=0.1 * z_bound_pos, normalize=quiver_normalize, arrow_length_ratio=0.4,
            cmap=cmap, norm=norm_metric, array=array,
            linewidth=0.8, alpha=0.9)

    # --- Visualise W₁, W₂, W₃ at the same quiver sample points ---------
    if show_W_basis_fields:
        # Re‑use the quiver cloud 'pts' if it exists; otherwise build a small fallback set
        if 'pts' in locals():
            basis_pts = pts
        else:
            rng_basis = npr.default_rng(17)
            N_basis   = 200
            r_vals    = (np.sqrt(np.sqrt(rng_basis.random(N_basis))) * s_max)
            theta_vals= rng_basis.uniform(0, 2*np.pi, N_basis)
            signs     = rng_basis.choice([-1, 1], size=N_basis)
            s_vals    = signs * r_vals
            alpha_vals= np.sqrt(2) * s_vals * np.cos(theta_vals)
            beta_vals = np.sqrt(2) * s_vals * np.sin(theta_vals)
            basis_pts = (
                s_vals[:,None]    * ell_unit +
                alpha_vals[:,None]* v1 +
                beta_vals[:,None] * v2
            )

        # Linear shear fields
        def W1(a, c, e): return np.array([0,      -a - c,  a + e])
        def W2(a, c, e): return np.array([-a - c,  0,      c + e])
        def W3(a, c, e): return np.array([c - e,   e - a,  a - c])

        colours = ['red', 'green', 'blue']        # distinguish W₁,W₂,W₃
        scale_factor_basis = 0.05                 # shrink arrows to 5 %
        for p in basis_pts:
            vecs_basis = [W1(*p), W2(*p), W3(*p)]
            for vec, col in zip(vecs_basis, colours):
                origin = _rot(p)
                tip    = _rot(p + scale_factor_basis * vec)
                dvec   = tip - origin
                ax.quiver(origin[0], origin[1], origin[2],
                          dvec[0], dvec[1], dvec[2],
                          color=col, linewidth=0.8,
                          alpha=0.8, arrow_length_ratio=0.09)

    return fig

# ==============================================================================
# Streamlit sidebar UI
# ==============================================================================
st.title("Hentsch Manifold Visualization")
st.sidebar.header("**Plot Controls**")
with st.sidebar.expander("**View Controls**", expanded=True):
    projection_type = st.selectbox("Projection", options=['persp', 'ortho'], index=0)
    elevation = st.slider("Elevation°", -90, 90, value=20, step=1)
    azimuth = st.slider("Azimuth°", -180, 180, value=-45, step=1)
    reorient_vertical = st.checkbox("𝓵-axis Vertical", value=False)

with st.sidebar.expander("**Framework**", expanded=False):
    st.markdown("### Axes")
    show_axes_dict = {}
    for k in "abcdef":
        show_axes_dict[k] = st.checkbox(f"axis {k}", value=True)
    st.markdown("### Complex Planes")
    show_planes = st.checkbox("Show Complex Planes", value=False)

with st.sidebar.expander("**Quadratic Cone**", expanded=True):
    show_cone = st.checkbox("Show Cone Surface", value=True)
    show_wireframe = st.checkbox("Show Cone Wireframe", value=True)
    # -------- Define the interaction handlers for the number box and slider
    def box_changed():
        # fetch whichever key is present; default to 0.0
        val = st.session_state.get("q_box", st.session_state.get("quadratic_value", 0.0))
        st.session_state["quadratic_value"] = val
        st.session_state["q_slider"] = val      # keep slider in sync
    def slider_changed():
        val = st.session_state.get("q_slider", st.session_state.get("quadratic_value", 0.0))
        st.session_state["quadratic_value"] = val
        st.session_state["q_box"] = val         # keep number box in sync
    # ----- widgets -----
    val = st.session_state.get("quadratic_value", 0.0)
    q_box = st.number_input(
        "Exact n value",
        -0.99, 0.99,
        value=val,
        key="q_box",
        step=0.001,
        format="%.3f",
        on_change=box_changed,
    )
    q_slider = st.slider(
        "Q = n Foliation Level",
        -0.99, 0.99,
        value=val,
        key="q_slider",
        step=0.001,
        format="%.3f",
        on_change=slider_changed,
    )
    quadratic_value = val

    scale_factor = st.slider("Scale Factor", 1.0, 100.0, value=1.0, step=0.01)
    show_screen_planes = st.checkbox("Show Screen Plane", value=False)
    s_value = st.slider("Screen Plane s-value", -1.0, 1.0, value=0.0, step=0.01)

with st.sidebar.expander("**Vector Fields**", expanded=True):
    st.markdown("### Parameters")
    theta_deg = st.slider("Shear Angle θ°", 0, 720, value=0, step=1)
    phi_deg   = st.slider("Shear Angle φ°", 0, 720, value=0, step=1)
    epsilon = st.slider("Shear Strength ε", -4.0, 4.0, value=3/(2*np.sqrt(6)), step=0.01)
    st.markdown("### Options")
    uniform_quiver_seed = st.checkbox("Use uniform quiver seeding", value=True)
    quiver_num = st.slider("Quiver Density", 12, 4320, value=2160, step=12)
    # show_vectors3d = st.checkbox("Show Field Vectors", value=True)
    quiver_normalize = st.checkbox("Normalize Quiver Vectors", value=False)
    upper_nape_only = st.checkbox("Show Upper Nape Only (ℓ ≥ 0)", value=False)
    use_radial_coloring = st.checkbox("Color by radial alignment", value=False)
    show_canonical_vector_field = st.checkbox("Canonical Field Vectors", value=False)
    show_shear_vector_field = st.checkbox("Shear Field Vectors", value=False)
    # Indented basis‑choice checkboxes
    col_pad, col_chk = st.columns([0.10, 0.90])
    with col_chk:
        show_basis = st.checkbox("Basis in Screen", value=False)
        show_W_basis_fields = st.checkbox("Basis at points", value=False)
    show_combined_vector_field = st.checkbox("Combined Deformation Vectors", value=False)


# ==============================================================================
# Plot and display
# ==============================================================================
# st.write("Rendering with ε =", epsilon, "and θ° =", theta_deg)
fig = make_plot(
    projection_type=projection_type,
    elevation=elevation,
    azimuth=azimuth,
    theta_deg=theta_deg,
    phi_deg=phi_deg,
    epsilon=epsilon,
    quadratic_value=quadratic_value,
    scale_factor=scale_factor,
    s_value=s_value,
    quiver_num=quiver_num,
    reorient_vertical=reorient_vertical,
    show_cone=show_cone,
    show_wireframe=show_wireframe,
    show_basis=show_basis,
    show_vectors3d=True,
    show_canonical_vector_field=show_canonical_vector_field,
    show_shear_vector_field=show_shear_vector_field,
    show_combined_vector_field=show_combined_vector_field,
    show_planes=show_planes,
    show_screen_planes=show_screen_planes,
    show_axes_dict=show_axes_dict,
    quiver_normalize=quiver_normalize,
    uniform_quiver_seed=uniform_quiver_seed,
    upper_nape_only=upper_nape_only,
    use_radial_coloring=use_radial_coloring,
)
if 'mag' in locals():
    # Optionally warn if there were invalid or extreme values
    if np.any(np.isnan(mag)) or np.any(np.isinf(mag)):
        st.warning("Plot contained invalid or extreme vector values that were sanitized.")

# ==============================================================================
# GIF Animation Controls and Generation
# ==============================================================================
import imageio.v2 as imageio
import tempfile

def generate_gif_animation(
    param_name,
    param_values,
    filename="animation.gif",
    duration=0.1,
    bounce=True
):
    with tempfile.TemporaryDirectory() as tmpdir:
        images = []
        progress = st.progress(0)
        status = st.empty()
        for i, val in enumerate(param_values):
            kwargs = {
                "projection_type": projection_type,
                "elevation": elevation,
                "azimuth": azimuth,
                "theta_deg": theta_deg,
                "phi_deg": phi_deg,
                "epsilon": epsilon,
                "quadratic_value": quadratic_value,
                "scale_factor": scale_factor,
                "s_value": s_value,
                "quiver_num": quiver_num,
                "reorient_vertical": reorient_vertical,
                "show_cone": show_cone,
                "show_wireframe": show_wireframe,
                "show_basis": show_basis,
                "show_vectors3d": True,
                "show_canonical_vector_field": show_canonical_vector_field,
                "show_shear_vector_field": show_shear_vector_field,
                "show_combined_vector_field": show_combined_vector_field,
                "show_planes": show_planes,
                "show_screen_planes": show_screen_planes,
                "show_axes_dict": show_axes_dict,
                "quiver_normalize": quiver_normalize,
                "uniform_quiver_seed": uniform_quiver_seed,
                "upper_nape_only": upper_nape_only,
                "use_radial_coloring": use_radial_coloring,
            }
            kwargs[param_name] = val
            fig = make_plot(**kwargs)
            img_path = os.path.join(tmpdir, f"frame_{i:03d}.png")
            fig.savefig(img_path)
            plt.close(fig)
            images.append(imageio.imread(img_path))
            progress.progress((i + 1) / len(param_values))
            status.text(f"Generating frame {i + 1} of {len(param_values)}")
        if bounce:
            images = images + images[-2::-1]
        imageio.mimsave(filename, images, duration=duration, loop=0)
        progress.empty()
        status.empty()
        return filename

# --- Function to convert GIF to MP4 using ffmpeg ---
def convert_gif_to_mp4(gif_path, mp4_path):
    cmd = [
        "ffmpeg",
        "-y",  # overwrite without asking
        "-i", gif_path,
        "-movflags", "faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        mp4_path,
    ]
    subprocess.run(cmd, check=True)

# --- Dual-parameter GIF animation function ---
def generate_gif_double_param(
    param1_name,
    param2_name,
    param1_values,
    param2_values,
    filename="animation_double.gif",
    duration=0.1,
    bounce=True
):
    with tempfile.TemporaryDirectory() as tmpdir:
        images = []
        progress = st.progress(0)
        status = st.empty()
        for i, (val1, val2) in enumerate(zip(param1_values, param2_values)):
            kwargs = {
                "projection_type": projection_type,
                "elevation": elevation,
                "azimuth": azimuth,
                "theta_deg": theta_deg,
                "phi_deg": phi_deg,
                "epsilon": epsilon,
                "quadratic_value": quadratic_value,
                "scale_factor": scale_factor,
                "s_value": s_value,
                "quiver_num": quiver_num,
                "reorient_vertical": reorient_vertical,
                "show_cone": show_cone,
                "show_wireframe": show_wireframe,
                "show_basis": show_basis,
                "show_vectors3d": True,
                "show_canonical_vector_field": show_canonical_vector_field,
                "show_shear_vector_field": show_shear_vector_field,
                "show_combined_vector_field": show_combined_vector_field,
                "show_planes": show_planes,
                "show_screen_planes": show_screen_planes,
                "show_axes_dict": show_axes_dict,
                "quiver_normalize": quiver_normalize,
                "uniform_quiver_seed": uniform_quiver_seed,
                "upper_nape_only": upper_nape_only,
                "use_radial_coloring": use_radial_coloring,
            }
            kwargs[param1_name] = val1
            kwargs[param2_name] = val2
            fig = make_plot(**kwargs)
            img_path = os.path.join(tmpdir, f"frame_{i:03d}.png")
            fig.savefig(img_path)
            plt.close(fig)
            images.append(imageio.imread(img_path))
            progress.progress((i + 1) / len(param1_values))
            status.text(f"Generating frame {i + 1} of {len(param1_values)}")
        if bounce:
            images = images + images[-2::-1]
        imageio.mimsave(filename, images, duration=duration, loop=0)
        progress.empty()
        status.empty()
        return filename

# ==============================================================================
# Streamlit GIF Animation UI
# ==============================================================================
with st.sidebar.expander("**GIF Animation**", expanded=False):
    bounce = st.checkbox("Bounce (ping-pong loop)", value=True)
    st.markdown("### Single-Parameter Animation")
    param_to_animate = st.selectbox(
        "Parameter to animate",
        ["theta_deg", "phi_deg", "epsilon", "quadratic_value", "elevation", "azimuth", "s_value"],
        index=0
    )
    anim_start = st.number_input("Start value", value=0.0, step=0.01)
    anim_end = st.number_input("End value", value=360.0, step=0.01)
    increments = st.number_input("Number of Increments", min_value=1, value=25, step=1)
    generate = st.button("Generate Animation")

    if generate:
        param_range = np.linspace(anim_start, anim_end, increments)
        output_file = generate_gif_animation(param_to_animate, param_range, filename="streamlit_animation.gif", bounce=bounce)
        with open(output_file, "rb") as f:
            st.download_button("Download GIF", f, file_name="streamlit_animation.gif", mime="image/gif")
        st.markdown(
            f'<img src="data:image/gif;base64,{base64.b64encode(open(output_file, "rb").read()).decode()}" '
            f'style="width:100%;" loop autoplay>',
            unsafe_allow_html=True
        )
        # Convert GIF to MP4 and offer download
        # mp4_file = output_file.replace(".gif", ".mp4")
        # convert_gif_to_mp4(output_file, mp4_file)
        # with open(mp4_file, "rb") as fmp4:
        #     st.download_button("Download MP4", fmp4, file_name="streamlit_animation.mp4", mime="video/mp4")

    # --- Dual-Parameter Animation UI ---
    st.markdown("### Dual-Parameter Animation")
    available_params = ["theta_deg", "phi_deg", "epsilon", "quadratic_value", "elevation", "azimuth", "s_value"]
    col1, col2 = st.columns(2)
    with col1:
        param1 = st.selectbox("Param 1", available_params, key="param1")
        p1_start = st.number_input("Start 1", value=0.0, key="p1_start")
        p1_end = st.number_input("End 1", value=360.0, key="p1_end")
    with col2:
        param2 = st.selectbox("Param 2", available_params, key="param2")
        p2_start = st.number_input("Start 2", value=0.0, key="p2_start")
        p2_end = st.number_input("End 2", value=1.0, key="p2_end")
    increments = st.number_input("Number of Increments", min_value=1, value=25, step=1, key="increments")
    generate2 = st.button("Generate Dual Animation")
    if generate2:
        p1_vals = np.linspace(p1_start, p1_end, increments)
        p2_vals = np.linspace(p2_start, p2_end, increments)
        param1_list = p1_vals
        param2_list = p2_vals
        output_file2 = generate_gif_double_param(param1, param2, param1_list, param2_list, filename="streamlit_dual_animation.gif", bounce=bounce)
        with open(output_file2, "rb") as f:
            st.download_button("Download Dual GIF", f, file_name="streamlit_dual_animation.gif", mime="image/gif")
        st.markdown(
            f'<img src="data:image/gif;base64,{base64.b64encode(open(output_file2, "rb").read()).decode()}" '
            f'style="width:100%;" loop autoplay>',
            unsafe_allow_html=True
        )
        # Convert dual GIF to MP4 and offer download
        # mp4_file2 = output_file2.replace(".gif", ".mp4")
        # convert_gif_to_mp4(output_file2, mp4_file2)
        # with open(mp4_file2, "rb") as fmp4_2:
        #     st.download_button("Download Dual MP4", fmp4_2, file_name="streamlit_dual_animation.mp4", mime="video/mp4")

st.pyplot(fig)
