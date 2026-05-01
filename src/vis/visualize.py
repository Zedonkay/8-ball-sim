#!/usr/bin/env python3
"""
SPH simulation visualizer.

Offline (default):
    python visualize.py sim.bin

Live (polls the file as the sim writes):
    python visualize.py sim.bin --live

Options:
    --no-ghosts hide ghost/wall particles
    --fps N     initial playback FPS (default 15). CLI accepts any finite N > 0 (no hard max).
                The on-screen FPS slider is limited to 0.01–240. Values above ~60 often do not
                actually play faster: Matplotlib’s GUI timer and the display refresh rate (commonly
                60 Hz) cap how often frames can be shown, especially for heavy 3D redraws.
"""

import argparse
import os
import re
import struct
import sys
import time
import numpy as np

# Pick an interactive backend before importing pyplot.
# On macOS the native 'macosx' backend works best; fall back to TkAgg.
import matplotlib
if sys.platform == "darwin":
    try:
        matplotlib.use("macosx")
    except Exception:
        matplotlib.use("TkAgg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers projection)
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.widgets import Button

# ── Sim constants (must match constants.h) ────────────────────────────────────
SPHERE_R = 0.15        # metres
FRAME_MAGIC_V1 = 0x464D4953  # "SIMF" little-endian uint32
FRAME_MAGIC_V2 = 0x464D4932  # "SIM2" little-endian uint32
GHOST_COLOR = "crimson"
FLUID_ALPHA = 0.1
GHOST_ALPHA = 0.25

# Header layout: magic(u32) step(i32) t(f32) n_fluid(i32) n_total(i32)
HEADER_FMT  = "<Iifii"
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 20 bytes
MAGIC_V1_BYTES = struct.pack("<I", FRAME_MAGIC_V1)
MAGIC_V2_BYTES = struct.pack("<I", FRAME_MAGIC_V2)


def _frame_interval_ms(fps):
    """
    Delay between animation ticks in milliseconds.

    FuncAnimation passes this to the GUI timer; keep a small positive floor so
    backends never get a zero interval (which would spin and ignore --fps).
    """
    if fps <= 0 or not np.isfinite(fps):
        raise ValueError("fps must be a positive finite number")
    return max(1.0, 1000.0 / float(fps))


def _set_animation_interval_ms(anim, interval_ms):
    """Apply a new frame interval to a running FuncAnimation."""
    anim.event_source.interval = interval_ms


def _render_sample(xs, ys, zs, values, max_points):
    if max_points is None or max_points <= 0 or len(xs) <= max_points:
        return xs, ys, zs, values
    stride = int(np.ceil(len(xs) / float(max_points)))
    return xs[::stride], ys[::stride], zs[::stride], values[::stride]


def _show_until_closed(fig):
    """
    Some macOS/Python backend combinations return from show() immediately even
    for interactive windows. Drive the GUI event loop explicitly until the user
    closes the figure.
    """
    plt.show(block=False)
    while plt.fignum_exists(fig.number):
        plt.pause(0.01)


def _load_vis_alpha_from_constants():
    """Load visual alpha constants from simulator/constants.h when available."""
    constants_path = os.path.join(
        os.path.dirname(__file__), "..", "simulator", "constants.h"
    )
    try:
        with open(constants_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return

    def _extract(name):
        pattern = rf"constexpr\s+float\s+{name}\s*=\s*([-+]?\d*\.?\d+)\s*f?\s*;"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    fluid_alpha = _extract("VIS_FLUID_ALPHA")
    ghost_alpha = _extract("VIS_GHOST_ALPHA")

    global FLUID_ALPHA, GHOST_ALPHA
    if fluid_alpha is not None:
        FLUID_ALPHA = fluid_alpha
    if ghost_alpha is not None:
        GHOST_ALPHA = ghost_alpha


_load_vis_alpha_from_constants()


# ── Binary I/O ────────────────────────────────────────────────────────────────

def _read_frame(f):
    """Read one frame from an open binary file.  Returns None on incomplete data."""
    header_bytes = f.read(HEADER_SIZE)
    if len(header_bytes) < HEADER_SIZE:
        return None
    magic, step, t, n_fluid, n_total = struct.unpack(HEADER_FMT, header_bytes)
    if magic not in (FRAME_MAGIC_V1, FRAME_MAGIC_V2):
        return None

    n_floats = n_total * 3 + n_fluid
    data_bytes = f.read(n_floats * 4)
    if len(data_bytes) < n_floats * 4:
        return None  # incomplete frame — caller should seek back

    arr = np.frombuffer(data_bytes, dtype=np.float32)
    pos_x   = arr[:n_total]
    pos_y   = arr[n_total : 2 * n_total]
    pos_z   = arr[2 * n_total : 3 * n_total]
    density = arr[3 * n_total :]

    die_pos = None
    die_quat = None
    die_half = None
    if magic == FRAME_MAGIC_V2:
        die_bytes = f.read(10 * 4)
        if len(die_bytes) < 10 * 4:
            return None
        die_arr = np.frombuffer(die_bytes, dtype=np.float32)
        die_pos = die_arr[0:3].copy()
        die_quat = die_arr[3:7].copy()   # w, x, y, z
        die_half = die_arr[7:10].copy()

    return dict(step=step, t=t, n_fluid=n_fluid, n_total=n_total,
                pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, density=density,
                die_pos=die_pos, die_quat=die_quat, die_half=die_half)


def _resync_to_magic(f, window_bytes=8192):
    """Seek to next potential frame magic for robust live tailing."""
    start = f.tell()
    chunk = f.read(window_bytes)
    if not chunk:
        f.seek(start)
        return False

    i1 = chunk.find(MAGIC_V1_BYTES)
    i2 = chunk.find(MAGIC_V2_BYTES)
    candidates = [i for i in (i1, i2) if i >= 0]
    if candidates:
        f.seek(start + min(candidates))
        return True

    # Keep a few bytes of overlap in case magic splits across reads.
    rewind = min(3, len(chunk))
    f.seek(start + len(chunk) - rewind)
    return False


def load_all_frames(path):
    frames = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            frame = _read_frame(f)
            if frame is None:
                f.seek(pos)  # leave file at last good position
                break
            frames.append(frame)
    return frames


# ── Sphere wireframe ──────────────────────────────────────────────────────────

def _sphere_lines(r=SPHERE_R, n=30):
    """Return (xs, ys, zs) lists for a wireframe sphere (latitude + longitude lines)."""
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    xs, ys, zs = [], [], []

    # latitude rings
    for vi in v:
        xs.append(r * np.sin(vi) * np.cos(u))
        ys.append(r * np.sin(vi) * np.sin(u))
        zs.append(np.full_like(u, r * np.cos(vi)))

    # longitude arcs
    for ui in u[::3]:
        xs.append(r * np.sin(v) * np.cos(ui))
        ys.append(r * np.sin(v) * np.sin(ui))
        zs.append(r * np.cos(v))

    return xs, ys, zs


def draw_sphere(ax, r=SPHERE_R):
    xs, ys, zs = _sphere_lines(r)
    for x, y, z in zip(xs, ys, zs):
        ax.plot(x, y, z, color="steelblue", alpha=0.15, linewidth=0.6)


def quat_to_rotmat(q_wxyz):
    w, x, y, z = q_wxyz
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-8:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def _box_edges():
    return [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def box_vertices_world(pos, quat_wxyz, half):
    hx, hy, hz = half
    local = np.array([
        [-hx, -hy, -hz],
        [ hx, -hy, -hz],
        [-hx,  hy, -hz],
        [ hx,  hy, -hz],
        [-hx, -hy,  hz],
        [ hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy,  hz],
    ], dtype=np.float32)
    rot = quat_to_rotmat(quat_wxyz)
    return local @ rot.T + pos


def die_face_labels_world(pos, quat_wxyz, half):
    """Return world-space die face centers, normals, and face numbers."""
    rot = quat_to_rotmat(quat_wxyz)
    hx, hy, hz = half
    # Opposite faces sum to 7; this mapping stays rigidly attached to local axes.
    local_faces = [
        (np.array([0.0, 0.0, hz], dtype=np.float32), "1"),   # +Z
        (np.array([0.0, hy, 0.0], dtype=np.float32), "2"),   # +Y
        (np.array([hx, 0.0, 0.0], dtype=np.float32), "3"),   # +X
        (np.array([-hx, 0.0, 0.0], dtype=np.float32), "4"),  # -X
        (np.array([0.0, -hy, 0.0], dtype=np.float32), "5"),  # -Y
        (np.array([0.0, 0.0, -hz], dtype=np.float32), "6"),  # -Z
    ]
    labels = []
    for local_center, text in local_faces:
        normal_local = local_center / (np.linalg.norm(local_center) + 1e-8)
        p_world = local_center @ rot.T + pos
        normal_world = normal_local @ rot.T
        labels.append((p_world, normal_world, text))
    return labels


def visible_die_face_labels(ax, pos, quat_wxyz, half):
    """Return only labels for die faces currently facing the camera."""
    elev = np.deg2rad(float(ax.elev))
    azim = np.deg2rad(float(ax.azim))
    camera_dir = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)],
        dtype=np.float32,
    )
    visible = []
    for p_world, normal_world, text in die_face_labels_world(pos, quat_wxyz, half):
        # In mplot3d this camera direction points from scene -> camera,
        # so front-facing outward normals have a negative dot product.
        if float(np.dot(normal_world, camera_dir)) < 0.0:
            visible.append((p_world, text))
    return visible


def density_z_stats(frame, z_threshold):
    nf = frame["n_fluid"]
    z = frame["pos_z"][:nf]
    density = frame["density"][:nf]
    low_mask = z < -z_threshold
    high_mask = z > z_threshold
    low_count = int(np.count_nonzero(low_mask))
    high_count = int(np.count_nonzero(high_mask))
    low_mean = float(np.mean(density[low_mask])) if low_count > 0 else float("nan")
    high_mean = float(np.mean(density[high_mask])) if high_count > 0 else float("nan")
    return low_mean, high_mean, low_count, high_count


# ── Figure setup ─────────────────────────────────────────────────────────────

def make_figure(show_density_vs_z=False):
    fig = plt.figure(figsize=(12, 8) if show_density_vs_z else (9, 8))
    if show_density_vs_z:
        ax = fig.add_axes([0.05, 0.12, 0.60, 0.85], projection="3d")
        ax_profile = fig.add_axes([0.70, 0.12, 0.27, 0.85])
    else:
        ax = fig.add_axes([0.05, 0.12, 0.85, 0.85], projection="3d")
        ax_profile = None
    ax.set_box_aspect([1, 1, 1])
    lim = SPHERE_R * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    draw_sphere(ax)
    if ax_profile is not None:
        ax_profile.set_xlabel("z (m)", fontsize=8)
        ax_profile.set_ylabel("density (kg/m^3)", fontsize=8)
        ax_profile.tick_params(labelsize=7)
        ax_profile.grid(alpha=0.2)
    return fig, ax, ax_profile


# ── Offline mode ──────────────────────────────────────────────────────────────

def run_offline(frames, show_ghosts, fps, stats_enabled, stats_every,
                z_threshold, show_density_vs_z):
    if not frames:
        sys.exit("No frames found in file.")

    fig, ax, ax_profile = make_figure(show_density_vs_z=show_density_vs_z)
    fig.suptitle("SPH Visualizer  —  drag to rotate", fontsize=10)

    # density range for colormap
    all_density = np.concatenate([f["density"] for f in frames])
    vmin, vmax = float(all_density.min()), float(all_density.max())
    if vmax == vmin:
        vmax = vmin + 1.0

    # initial scatter objects
    initial_idx = 0
    if show_ghosts:
        for idx, fr in enumerate(frames):
            if fr["n_total"] > fr["n_fluid"]:
                initial_idx = idx
                break
    f0 = frames[initial_idx]
    scatter_fluid = ax.scatter(
        f0["pos_x"][:f0["n_fluid"]],
        f0["pos_y"][:f0["n_fluid"]],
        f0["pos_z"][:f0["n_fluid"]],
        c=f0["density"], cmap="viridis",
        vmin=vmin, vmax=vmax,
        s=50, alpha=FLUID_ALPHA, depthshade=True, zorder=3
    )

    scatter_ghost = None
    if show_ghosts and f0["n_total"] > f0["n_fluid"]:
        ng = f0["n_total"] - f0["n_fluid"]
        scatter_ghost = ax.scatter(
            f0["pos_x"][f0["n_fluid"]:f0["n_total"]],
            f0["pos_y"][f0["n_fluid"]:f0["n_total"]],
            f0["pos_z"][f0["n_fluid"]:f0["n_total"]],
            c=GHOST_COLOR, s=4, alpha=GHOST_ALPHA, depthshade=False
        )

    die_lines = []
    die_texts = []
    edges = _box_edges()
    if f0["die_pos"] is not None and f0["die_quat"] is not None and f0["die_half"] is not None:
        verts = box_vertices_world(f0["die_pos"], f0["die_quat"], f0["die_half"])
        for a, b in edges:
            ln, = ax.plot(
                [verts[a, 0], verts[b, 0]],
                [verts[a, 1], verts[b, 1]],
                [verts[a, 2], verts[b, 2]],
                color="black",
                linewidth=1.0,
                alpha=0.8,
            )
            die_lines.append(ln)
        for p_world, text in visible_die_face_labels(ax, f0["die_pos"], f0["die_quat"], f0["die_half"]):
            die_texts.append(
                ax.text(
                    p_world[0], p_world[1], p_world[2], text,
                    color="black", fontsize=8, ha="center", va="center"
                )
            )

    cbar = fig.colorbar(scatter_fluid, ax=ax, pad=0.02, shrink=0.6, label="density (kg/m³)")
    title = ax.set_title(f"step {f0['step']}   t = {f0['t']:.4f} s", fontsize=9)
    profile_scatter = None
    if show_density_vs_z and ax_profile is not None:
        nf0 = f0["n_fluid"]
        profile_scatter = ax_profile.scatter(
            f0["pos_z"][:nf0], f0["density"][:nf0], s=6, alpha=0.35, c="teal"
        )
        ax_profile.set_xlim(-SPHERE_R, SPHERE_R)
        ax_profile.set_ylim(vmin, vmax)

    # sliders (stacked: FPS above frame index so both stay usable)
    ax_fps = fig.add_axes([0.12, 0.048, 0.62, 0.026])
    smin, smax = 0.01, 240.0
    sinit = float(min(max(fps, smin), smax))
    fps_slider = Slider(
        ax_fps, "FPS", smin, smax, valinit=sinit, valstep=0.05, color="steelblue"
    )

    ax_slider = fig.add_axes([0.12, 0.015, 0.62, 0.026])
    slider = Slider(ax_slider, "frame", 0, len(frames) - 1,
                    valinit=initial_idx, valstep=1, color="steelblue")

    playing = [True]
    ax_btn = fig.add_axes([0.77, 0.028, 0.11, 0.042])
    pause_btn = Button(ax_btn, "Pause")

    # Auto-play index — keep in sync with the frame slider (scrub, keys, animate).
    frame_idx = [initial_idx]

    def update_frame(idx):
        nonlocal scatter_ghost, profile_scatter, die_lines, die_texts
        idx = int(idx)
        frame_idx[0] = idx
        fr = frames[idx]
        nf = fr["n_fluid"]

        scatter_fluid._offsets3d = (
            fr["pos_x"][:nf],
            fr["pos_y"][:nf],
            fr["pos_z"][:nf],
        )
        scatter_fluid.set_array(fr["density"])

        if show_ghosts:
            if fr["n_total"] > nf:
                gx = fr["pos_x"][nf:fr["n_total"]]
                gy = fr["pos_y"][nf:fr["n_total"]]
                gz = fr["pos_z"][nf:fr["n_total"]]
                if scatter_ghost is None:
                    scatter_ghost = ax.scatter(
                        gx, gy, gz, c=GHOST_COLOR, s=4, alpha=GHOST_ALPHA, depthshade=False
                    )
                else:
                    scatter_ghost._offsets3d = (gx, gy, gz)
                    scatter_ghost.set_visible(True)
            elif scatter_ghost is not None:
                scatter_ghost.set_visible(False)

        if fr["die_pos"] is not None and fr["die_quat"] is not None and fr["die_half"] is not None:
            verts = box_vertices_world(fr["die_pos"], fr["die_quat"], fr["die_half"])
            if not die_lines:
                for a, b in edges:
                    ln, = ax.plot(
                        [verts[a, 0], verts[b, 0]],
                        [verts[a, 1], verts[b, 1]],
                        [verts[a, 2], verts[b, 2]],
                        color="black",
                        linewidth=1.0,
                        alpha=0.8,
                    )
                    die_lines.append(ln)
            else:
                for ln, (a, b) in zip(die_lines, edges):
                    ln.set_data_3d(
                        [verts[a, 0], verts[b, 0]],
                        [verts[a, 1], verts[b, 1]],
                        [verts[a, 2], verts[b, 2]],
                    )
                for ln in die_lines:
                    ln.set_visible(True)
            for txt in die_texts:
                txt.remove()
            die_texts = []
            for p_world, text in visible_die_face_labels(ax, fr["die_pos"], fr["die_quat"], fr["die_half"]):
                die_texts.append(
                    ax.text(
                        p_world[0], p_world[1], p_world[2], text,
                        color="black", fontsize=8, ha="center", va="center"
                    )
                )
        else:
            for ln in die_lines:
                ln.set_visible(False)
            for txt in die_texts:
                txt.remove()
            die_texts = []

        if show_density_vs_z and ax_profile is not None and profile_scatter is not None:
            profile_scatter.set_offsets(np.column_stack((fr["pos_z"][:nf], fr["density"][:nf])))
            y_min = float(fr["density"].min())
            y_max = float(fr["density"].max())
            if y_max <= y_min:
                y_max = y_min + 1.0
            ax_profile.set_ylim(y_min, y_max)

        if stats_enabled and (fr["step"] % max(1, stats_every) == 0):
            low_mean, high_mean, low_count, high_count = density_z_stats(fr, z_threshold)
            print(
                f"[density-z] step={fr['step']:6d} t={fr['t']:.5f}s  "
                f"rho(z<-{z_threshold:.3f})={low_mean:.2f} (n={low_count})  "
                f"rho(z>{z_threshold:.3f})={high_mean:.2f} (n={high_count})"
            )

        title.set_text(f"step {fr['step']}   t = {fr['t']:.4f} s   "
                       f"[{idx+1}/{len(frames)}]")
        fig.canvas.draw_idle()

    slider.on_changed(update_frame)

    def animate(_):
        if playing[0]:
            frame_idx[0] = (frame_idx[0] + 1) % len(frames)
            slider.set_val(frame_idx[0])  # triggers update_frame

    anim = FuncAnimation(
        fig, animate, interval=_frame_interval_ms(fps), cache_frame_data=False
    )

    def on_fps_change(val):
        _set_animation_interval_ms(anim, _frame_interval_ms(float(val)))

    fps_slider.on_changed(on_fps_change)

    def toggle_pause(_event):
        playing[0] = not playing[0]
        pause_btn.label.set_text("Pause" if playing[0] else "Play")
        fig.canvas.draw_idle()

    pause_btn.on_clicked(toggle_pause)

    def on_key(event):
        if event.key == " ":
            playing[0] = not playing[0]
            pause_btn.label.set_text("Pause" if playing[0] else "Play")
        elif event.key == "right":
            frame_idx[0] = min(frame_idx[0] + 1, len(frames) - 1)
            slider.set_val(frame_idx[0])
        elif event.key == "left":
            frame_idx[0] = max(frame_idx[0] - 1, 0)
            slider.set_val(frame_idx[0])

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.text(
        0.01,
        0.002,
        "Button/Space: play-pause   ←/→: step   FPS slider: speed   drag: rotate   scroll: zoom",
        fontsize=7,
        color="gray",
    )

    _show_until_closed(fig)


# ── Live mode ─────────────────────────────────────────────────────────────────

def run_live(path, show_ghosts, fps, stats_enabled, stats_every,
             z_threshold, show_density_vs_z, live_latest,
             max_render_particles):
    fig, ax, ax_profile = make_figure(show_density_vs_z=show_density_vs_z)
    fig.suptitle("SPH Visualizer  —  LIVE", fontsize=10)

    file_handle = [None]
    byte_offset  = [0]
    scatter_fluid  = [None]
    scatter_ghost  = [None]
    title_obj      = [ax.set_title("waiting for first frame…", fontsize=9)]
    cbar_obj       = [None]
    last_vmin      = [None]
    last_vmax      = [None]
    profile_scatter = [None]
    die_lines = []
    die_texts = []
    edges = _box_edges()

    def _open_file():
        try:
            fh = open(path, "rb")
            file_handle[0] = fh
        except OSError:
            pass

    _open_file()

    def animate(_):
        fh = file_handle[0]
        if fh is None:
            _open_file()
            return

        pos = fh.tell()
        fr = _read_frame(fh)
        if fr is None:
            fh.seek(pos)
            # If the stream is misaligned/corrupted (e.g., writer restarted),
            # attempt to recover by scanning for the next frame magic.
            _resync_to_magic(fh)
            return
        if live_latest:
            while True:
                next_pos = fh.tell()
                next_fr = _read_frame(fh)
                if next_fr is None:
                    fh.seek(next_pos)
                    break
                fr = next_fr

        nf = fr["n_fluid"]
        density = fr["density"]

        vmin = float(density.min())
        vmax = float(density.max())
        if vmax == vmin:
            vmax = vmin + 1.0

        px = fr["pos_x"][:nf]
        py = fr["pos_y"][:nf]
        pz = fr["pos_z"][:nf]
        px_r, py_r, pz_r, density_r = _render_sample(
            px, py, pz, density, max_render_particles
        )

        if scatter_fluid[0] is None:
            scatter_fluid[0] = ax.scatter(
                px_r, py_r, pz_r,
                c=density_r, cmap="viridis", vmin=vmin, vmax=vmax,
                s=50, alpha=FLUID_ALPHA, depthshade=True, zorder=3
            )
            cbar_obj[0] = fig.colorbar(scatter_fluid[0], ax=ax,
                                       pad=0.02, shrink=0.6, label="density (kg/m³)")
            last_vmin[0], last_vmax[0] = vmin, vmax
            if show_density_vs_z and ax_profile is not None:
                profile_scatter[0] = ax_profile.scatter(
                    pz, density, s=6, alpha=0.35, c="teal"
                )
                ax_profile.set_xlim(-SPHERE_R, SPHERE_R)
                ax_profile.set_ylim(vmin, vmax)
        else:
            scatter_fluid[0]._offsets3d = (px_r, py_r, pz_r)
            scatter_fluid[0].set_array(density_r)
            if vmin != last_vmin[0] or vmax != last_vmax[0]:
                scatter_fluid[0].set_clim(vmin, vmax)
                last_vmin[0], last_vmax[0] = vmin, vmax
            if show_density_vs_z and ax_profile is not None and profile_scatter[0] is not None:
                profile_scatter[0].set_offsets(np.column_stack((pz, density)))
                ax_profile.set_ylim(vmin, vmax)

        if show_ghosts:
            if fr["n_total"] > nf:
                gx = fr["pos_x"][nf:fr["n_total"]]
                gy = fr["pos_y"][nf:fr["n_total"]]
                gz = fr["pos_z"][nf:fr["n_total"]]
                if scatter_ghost[0] is None:
                    scatter_ghost[0] = ax.scatter(gx, gy, gz, c=GHOST_COLOR,
                                                  s=4, alpha=GHOST_ALPHA, depthshade=False)
                else:
                    scatter_ghost[0]._offsets3d = (gx, gy, gz)
                    scatter_ghost[0].set_visible(True)
            elif scatter_ghost[0] is not None:
                scatter_ghost[0].set_visible(False)

        if fr["die_pos"] is not None and fr["die_quat"] is not None and fr["die_half"] is not None:
            verts = box_vertices_world(fr["die_pos"], fr["die_quat"], fr["die_half"])
            if not die_lines:
                for a, b in edges:
                    ln, = ax.plot(
                        [verts[a, 0], verts[b, 0]],
                        [verts[a, 1], verts[b, 1]],
                        [verts[a, 2], verts[b, 2]],
                        color="black",
                        linewidth=1.0,
                        alpha=0.8,
                    )
                    die_lines.append(ln)
            else:
                for ln, (a, b) in zip(die_lines, edges):
                    ln.set_data_3d(
                        [verts[a, 0], verts[b, 0]],
                        [verts[a, 1], verts[b, 1]],
                        [verts[a, 2], verts[b, 2]],
                    )
                for ln in die_lines:
                    ln.set_visible(True)
            for txt in die_texts:
                txt.remove()
            die_texts[:] = []
            for p_world, text in visible_die_face_labels(ax, fr["die_pos"], fr["die_quat"], fr["die_half"]):
                die_texts.append(
                    ax.text(
                        p_world[0], p_world[1], p_world[2], text,
                        color="black", fontsize=8, ha="center", va="center"
                    )
                )
        else:
            for ln in die_lines:
                ln.set_visible(False)
            for txt in die_texts:
                txt.remove()
            die_texts[:] = []

        if stats_enabled and (fr["step"] % max(1, stats_every) == 0):
            low_mean, high_mean, low_count, high_count = density_z_stats(fr, z_threshold)
            print(
                f"[density-z] step={fr['step']:6d} t={fr['t']:.5f}s  "
                f"rho(z<-{z_threshold:.3f})={low_mean:.2f} (n={low_count})  "
                f"rho(z>{z_threshold:.3f})={high_mean:.2f} (n={high_count})"
            )

        title_obj[0].set_text(f"step {fr['step']}   t = {fr['t']:.4f} s")
        fig.canvas.draw_idle()

    anim = FuncAnimation(
        fig, animate, interval=_frame_interval_ms(fps), cache_frame_data=False
    )

    ax_fps = fig.add_axes([0.12, 0.03, 0.62, 0.03])
    smin, smax = 0.01, 240.0
    sinit = float(min(max(fps, smin), smax))
    fps_slider = Slider(
        ax_fps, "FPS", smin, smax, valinit=sinit, valstep=0.05, color="steelblue"
    )

    def on_fps_change(val):
        _set_animation_interval_ms(anim, _frame_interval_ms(float(val)))

    fps_slider.on_changed(on_fps_change)

    fig.text(
        0.01,
        0.005,
        "drag plot: rotate   scroll: zoom   FPS slider: playback speed (async with writer)",
        fontsize=7,
        color="gray",
    )
    _show_until_closed(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_fps_arg(s):
    v = float(s)
    if not (v > 0) or not np.isfinite(v):
        raise argparse.ArgumentTypeError("fps must be a positive finite number")
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="path to sim.bin output file")
    parser.add_argument("--live",   action="store_true",
                        help="live mode: poll file while sim is running")
    parser.set_defaults(ghosts=True)
    parser.add_argument("--ghosts", dest="ghosts", action="store_true",
                        help="show ghost/wall particles (default)")
    parser.add_argument("--no-ghosts", dest="ghosts", action="store_false",
                        help="hide ghost/wall particles")
    parser.add_argument(
        "--fps",
        type=_parse_fps_arg,
        default=15,
        help=(
            "initial playback FPS (default 15); any value > 0. Slider range is 0.01–240. "
            "Rates above ~60 Hz are often not reached (GUI timer / vsync / 3D draw cost)."
        ),
    )
    parser.add_argument("--density-z-stats", action="store_true",
                        help="print mean density for z<-threshold vs z>threshold")
    parser.add_argument("--density-z-threshold", type=float, default=0.05,
                        help="z threshold in meters for density-z stats (default 0.05)")
    parser.add_argument("--density-z-stats-every", type=int, default=1,
                        help="print density-z stats every N simulation steps (default 1)")
    parser.add_argument("--plot-density-vs-z", action="store_true",
                        help="show density-vs-z scatter subplot")
    parser.add_argument("--live-latest", action="store_true",
                        help="in live mode, drain queued frames and render the newest complete frame")
    parser.add_argument("--max-render-particles", type=int, default=0,
                        help="render at most N fluid particles in live mode (0 = all)")
    args = parser.parse_args()

    if args.live:
        run_live(args.file, args.ghosts, args.fps,
                 args.density_z_stats, args.density_z_stats_every,
                 args.density_z_threshold, args.plot_density_vs_z,
                 args.live_latest, args.max_render_particles)
    else:
        print(f"Loading {args.file} …")
        frames = load_all_frames(args.file)
        print(f"Loaded {len(frames)} frames.")
        run_offline(frames, args.ghosts, args.fps,
                    args.density_z_stats, args.density_z_stats_every,
                    args.density_z_threshold, args.plot_density_vs_z)


if __name__ == "__main__":
    main()
