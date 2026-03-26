import numpy as np
import tifffile
import os
import glob
from tkinter import Tk, simpledialog, messagebox
from tkinter import filedialog
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import UnivariateSpline

"""
Surface flattening via CIDI cost map + Distance Bias + Dynamic Programming + anchor constraints.

Why this approach:
  The target surface is characterised by a LOW→HIGH intensity transition along Z
  (strong backscattering object, low axial resolution). The surface boundary is
  often blurry (low contrast), and multiple parallel scattering layers exist above
  and below the true surface — all with nearly identical Z-shape. This means:
    • Simple gradient-peak detection is ambiguous: every layer boundary is a valid peak.
    • The "best" edge by image contrast may not be the target layer (user may
      deliberately click a higher-contrast neighbouring layer to use as proxy).

  NEW in v4 — Distance-to-anchor bias:
    When there are multiple rising-edge candidates in a column, we prefer the one
    that is CLOSEST (in Z) to the user's anchor point, on the SAME SIDE as the
    anchor. This is implemented by adding a soft penalty term to the CIDI cost:

        total_cost(z) = cidi_cost(z) + ANCHOR_BIAS * |z - z_anchor_interpolated|

    where z_anchor_interpolated is the linearly interpolated anchor Z for each column.
    This bias steers the DP path toward the layer the user pointed at, without
    removing the freedom to deviate slightly where the image signal demands it.
    ANCHOR_BIAS controls the strength: 0 = pure CIDI (v3 behaviour), larger values
    = stronger pull toward the anchor Z.

  This implementation:
    1. Builds a per-column COST MAP (CIDI): cost is LOWEST at positions where
       intensity transitions sharply from LOW (outside) to HIGH (inside the object).
    2. Adds a DISTANCE BIAS to the cost map, centred on the interpolated anchor Z.
       Multiple parallel edges become distinguishable — the one closest to the
       user's click is preferred.
    3. Solves for the globally optimal SURFACE PATH using Dynamic Programming (DP)
       with a smoothness penalty that prevents jumping between parallel layers.
    4. Hard ANCHOR CONSTRAINTS at clicked columns ensure the path cannot deviate
       more than ±ANCHOR_TOL pixels from the exact click Z.

Tunable parameters (top of file):
  HALF_WIN     : local Z search window half-width around each anchor (px). Default 7.
  SMOOTH_SIG   : Gaussian sigma for Z-profile smoothing before CIDI.       Default 1.
  DP_LAMBDA    : DP smoothness penalty weight λ·|Δz|².                     Default 10.
  ANCHOR_TOL   : Hard constraint radius around each anchor click (px).     Default 10.
  ANCHOR_BIAS  : Distance-to-anchor bias weight (cost units / px).         Default 0.3.
                 Increase if DP still picks the wrong parallel layer.
                 Set to 0 to revert to v3 behaviour.
  SPLINE_S     : UnivariateSpline smoothing factor (None = auto).
  N_CLICKS     : Anchor clicks per plane.                                   Default 7.
"""

# ─── Tunable parameters ───────────────────────────────────────────────────────
HALF_WIN    = 7      # Local Z search window = clicked_z ± HALF_WIN
SMOOTH_SIG  = 1.0   # Gaussian sigma for Z-profile smoothing
DP_LAMBDA   = 10.0  # DP smoothness penalty: λ * |Δz|^2 per adjacent column.
# 控制路径的平滑程度，如果检测到的曲线在列间跳动，应当增大
ANCHOR_TOL  = 10    # Hard DP constraint radius around each anchor (px).
# 控制锚点约束的松紧，如果用户点击精度有限可以适当放大，但如果层间距很小则需要收紧。
ANCHOR_BIAS = 0.3   # Distance-to-anchor bias weight (cost units / px).
# 当窗口内有多条平行边缘时，控制对"离锚点最近的那条"的偏好强度。
# 0 = 纯CIDI（等价于v3行为）; 增大 = 更强地贴近用户点击Z位置。
# 建议范围: 0.1（弱偏置）~ 1.0（强偏置）。
SPLINE_S    = None  # UnivariateSpline smoothing (None = auto, 0 = exact)
N_CLICKS    = 7     # Anchor clicks per plane
OFFSET_SMOOTH_SIG = 5.0  # 2-D Gaussian sigma for final offset map smoothing (px).
# 消除逐行DP产生的X方向条纹，同时平滑Y方向跳变。
# 增大 = 更平滑但细节损失；减小 = 保留更多局部细节。建议范围 3~15。
Is_RegistUsingOffsetMap = False  # True  → 跳过手动选点，直接读取目标文件夹下的 offset_map.npy 进行三维矫正。
# False → 正常流程：手动选点 → DP检测 → 确认 → 矫正。
# ──────────────────────────────────────────────────────────────────────────────


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_full_tiffstack(path):
    with tifffile.TiffFile(path) as tif:
        num_pages = len(tif.pages)
        volume = tif.asarray(key=range(num_pages), out='memmap')
    return volume


def volume_registor(volume_path, offSetMap):
    volume = load_full_tiffstack(volume_path)
    try:
        registered = fast_roll_along_z(volume, offSetMap)
        tifffile.imwrite(volume_path, registered)
        print(f"  Registered: {volume_path}")
    except (ValueError, AssertionError) as e:
        raise ValueError(f"Shape mismatch for {volume_path}: {e}")


# ─── Core registration ────────────────────────────────────────────────────────

def fast_roll_along_z(volume, offSet_map):
    """Vectorised circular roll along Z axis. Shape: (dim_y, dim_z, dim_x)."""
    dim_y, dim_z, dim_x = volume.shape[:3]
    assert offSet_map.shape == (dim_y, dim_x), \
        f"offSetMap {offSet_map.shape} != volume YX ({dim_y},{dim_x})"
    z_indices     = np.arange(dim_z).reshape(1, dim_z, 1)
    new_z_indices = (z_indices - offSet_map[:, None, :]) % dim_z
    return volume[
        np.arange(dim_y)[:, None, None],
        new_z_indices,
        np.arange(dim_x)[None, None, :]
    ]


# ─── CIDI cost map ────────────────────────────────────────────────────────────

def cidi_cost_column(profile_1d, z_lo, z_hi, smooth_sig,
                     z_anchor=None, anchor_bias=0.0):
    """
    Compute a per-Z cost vector for one column profile within [z_lo, z_hi].

    CIDI logic:
      For each candidate surface Z = s, we want the intensity to be LOW above s
      and HIGH below s (i.e. a rising edge when going from outside to inside).

        cidi_cost(s) = mean(I[above s]) - mean(I[below s])

      This is MINIMISED at positions where intensity jumps from low to high.
      w is chosen adaptively as half the local window width.

    Distance-to-anchor bias (NEW in v4):
      When multiple parallel rising edges exist in the column, their CIDI costs
      are nearly equal and DP cannot distinguish them reliably.  We break this
      tie by adding a soft penalty proportional to the Z distance from the
      interpolated anchor position:

        total_cost(s) = cidi_cost(s) + anchor_bias * |s_global - z_anchor|

      where s_global = z_lo + s is the absolute Z index of candidate s.
      This steers the DP toward the edge CLOSEST to the user's click, regardless
      of which parallel layer has marginally better image contrast.

    Parameters
    ----------
    profile_1d  : 1-D float array, full Z column
    z_lo, z_hi  : search window bounds (absolute Z indices)
    smooth_sig  : Gaussian sigma for pre-smoothing
    z_anchor    : interpolated anchor Z for this column (absolute). None = no bias.
    anchor_bias : weight of the distance-to-anchor penalty (cost units / px).

    Returns
    -------
    cost : float array, length = z_hi - z_lo + 1  (local indices, 0 = z_lo)
    """
    dim_z   = len(profile_1d)
    z_lo    = max(0, z_lo)
    z_hi    = min(dim_z - 1, z_hi)
    win_len = z_hi - z_lo + 1

    if win_len < 4:
        return np.zeros(win_len)

    local = gaussian_filter1d(
        profile_1d[z_lo: z_hi + 1].astype(np.float32), sigma=smooth_sig
    )
    # half-window for the above/below averaging
    w    = max(2, win_len // 6)
    cost = np.zeros(win_len, dtype=np.float32)

    for s in range(win_len):
        # pixels BELOW s (inside the object): should be high
        lo_start = max(0, s - w)
        below    = local[lo_start: s].mean() if s > lo_start else local[s]
        # pixels ABOVE s (outside the object): should be low
        hi_end   = min(win_len, s + w)
        above    = local[s: hi_end].mean() if hi_end > s else local[s]
        # CIDI cost: low where below is bright and above is dark
        cost[s]  = above - below

    # ── Distance-to-anchor bias ──────────────────────────────────────────────
    # Add a term proportional to |absolute_z - z_anchor|.
    # This makes the cost bowl tilt toward the anchor Z so that, among
    # equally-good CIDI candidates, the closest one wins.
    if z_anchor is not None and anchor_bias > 0.0:
        abs_z    = np.arange(z_lo, z_hi + 1, dtype=np.float32)
        cost    += anchor_bias * np.abs(abs_z - z_anchor)

    return cost.astype(np.float64)


# ─── DP surface detection ─────────────────────────────────────────────────────

def dp_surface_detection(slice_2d, anchors, half_win, smooth_sig,
                         dp_lambda, anchor_tol, anchor_bias=ANCHOR_BIAS):
    """
    Detect the surface curve in a 2-D slice (shape: [dim_z, dim_x]) using
    CIDI cost map + distance-to-anchor bias + Dynamic Programming + anchor hard constraints.

    Parameters
    ----------
    slice_2d     : 2-D array, shape (dim_z, dim_x)
    anchors      : list of (x_click, z_click) tuples from user
    half_win     : local Z-search half-width (px)
    smooth_sig   : Gaussian sigma for CIDI
    dp_lambda    : smoothness penalty weight
    anchor_tol   : radius of hard constraint at anchor columns
    anchor_bias  : distance-to-anchor bias weight (cost units / px).
                   When >0, prefers the edge closest to the interpolated anchor Z.

    Returns
    -------
    surface_z  : 1-D int array, length dim_x — surface Z for every column
    cost_map   : 2-D float array, shape (dim_z, dim_x) — visualisation aid
    """
    dim_z, dim_x = slice_2d.shape

    # ── 1. Build anchor dictionary: x_col → (z_click, z_lo, z_hi) ──
    anchor_dict = {}
    for (xc, zc) in anchors:
        xc = int(np.clip(xc, 0, dim_x - 1))
        zc = int(np.clip(zc, 0, dim_z - 1))
        anchor_dict[xc] = (zc,
                           max(0,        zc - half_win),
                           min(dim_z - 1, zc + half_win))

    # ── 2. Interpolate z_lo / z_hi / z_anchor for every column from anchors ──
    if not anchor_dict:
        raise ValueError("No anchors provided.")

    ax_sorted  = sorted(anchor_dict.keys())
    z_centres  = np.array([anchor_dict[x][0] for x in ax_sorted], dtype=np.float64)
    ax_arr     = np.array(ax_sorted, dtype=np.float64)
    all_x      = np.arange(dim_x, dtype=np.float64)
    z_c_interp = np.interp(all_x, ax_arr, z_centres)   # interpolated anchor Z per column
    z_lo_all   = np.clip(z_c_interp - half_win, 0, dim_z - 1).astype(int)
    z_hi_all   = np.clip(z_c_interp + half_win, 0, dim_z - 1).astype(int)

    # ── 3. Build full cost map (dim_z, dim_x) with distance bias ──
    # NaN outside the local window; bias tilts costs toward z_c_interp[x].
    cost_map = np.full((dim_z, dim_x), np.nan, dtype=np.float64)
    for x in range(dim_x):
        zl, zh = z_lo_all[x], z_hi_all[x]
        col_cost = cidi_cost_column(
            slice_2d[:, x], zl, zh, smooth_sig,
            z_anchor=z_c_interp[x],   # ← pass interpolated anchor Z
            anchor_bias=anchor_bias    # ← bias weight
        )
        cost_map[zl: zh + 1, x] = col_cost

    # ── 4. Dynamic Programming ──
    #    State: surface_z[x] ∈ [z_lo_all[x], z_hi_all[x]]
    #    Transition: cost(x, z) + λ·(z - z_prev)²
    #    Anchor constraint: at anchor columns, restrict z to [zc-tol, zc+tol]

    INF = 1e18

    # dp_val[z] = minimum total cost to reach column x with surface at z
    # dp_prev[z] = z of previous column that achieved dp_val[z]

    # Initialise at first column
    x0   = 0
    zl0, zh0 = z_lo_all[x0], z_hi_all[x0]
    n0   = zh0 - zl0 + 1
    dp_val  = np.full(dim_z, INF)
    dp_back = np.zeros(dim_z, dtype=int)

    # Apply anchor constraint at x=0 if present
    if x0 in anchor_dict:
        zc, _, _ = anchor_dict[x0]
        lo_c = max(zl0, zc - anchor_tol)
        hi_c = min(zh0, zc + anchor_tol)
    else:
        lo_c, hi_c = zl0, zh0

    for z in range(lo_c, hi_c + 1):
        dp_val[z]  = cost_map[z, x0] if not np.isnan(cost_map[z, x0]) else INF
        dp_back[z] = z

    # Traceback storage
    traceback = np.zeros((dim_x, dim_z), dtype=np.int32)
    traceback[0, :] = np.arange(dim_z, dtype=np.int32)

    for x in range(1, dim_x):
        zl, zh = z_lo_all[x], z_hi_all[x]

        # Anchor constraint for this column
        if x in anchor_dict:
            zc, _, _ = anchor_dict[x]
            lo_c = max(zl, zc - anchor_tol)
            hi_c = min(zh, zc + anchor_tol)
        else:
            lo_c, hi_c = zl, zh

        new_dp_val  = np.full(dim_z, INF)
        new_dp_back = np.zeros(dim_z, dtype=int)

        # For efficiency: precompute prefix-min of (dp_val + λ*z²) and suffix-min
        # to allow O(W) DP per column instead of O(W²).
        # (Standard SMAWK / divide-and-conquer would be O(W log W), but W ≤ 2*HALF_WIN
        # is small enough that the O(W²) scan is fast in practice.)
        prev_z_range = np.where(dp_val < INF)[0]
        if len(prev_z_range) == 0:
            break

        z_prev_min = prev_z_range[0]
        z_prev_max = prev_z_range[-1]

        for z in range(lo_c, hi_c + 1):
            c = cost_map[z, x]
            if np.isnan(c):
                continue
            # Find best previous z: minimise dp_val[zp] + λ*(z - zp)²
            # Restrict zp search to [z_prev_min, z_prev_max] for speed
            zp_lo = max(z_prev_min, z - half_win)
            zp_hi = min(z_prev_max, z + half_win)
            zp_range = np.arange(zp_lo, zp_hi + 1)
            candidates = dp_val[zp_range] + dp_lambda * (z - zp_range) ** 2
            best_idx   = int(np.argmin(candidates))
            best_zp    = zp_lo + best_idx
            new_dp_val[z]  = c + candidates[best_idx]
            new_dp_back[z] = best_zp

        dp_val  = new_dp_val
        dp_back = new_dp_back
        traceback[x, :] = dp_back

    # ── 5. Traceback ──
    # Find the best final z
    valid_mask = dp_val < INF
    if not np.any(valid_mask):
        # Fallback: use interpolated z centres
        surface_z = z_c_interp.astype(int)
        return surface_z, cost_map

    best_z_final = int(np.argmin(np.where(valid_mask, dp_val, INF)))

    surface_z    = np.zeros(dim_x, dtype=int)
    surface_z[-1] = best_z_final
    for x in range(dim_x - 2, -1, -1):
        surface_z[x] = traceback[x + 1, surface_z[x + 1]]

    return surface_z, cost_map


# ─── Spline smoothing of the detected surface curve ───────────────────────────

def smooth_surface_spline(surface_z, full_dim, s=SPLINE_S):
    """
    Apply optional UnivariateSpline smoothing to the DP-detected surface curve.

    Returns smoothed_z as float array of length full_dim.
    """
    x = np.arange(full_dim, dtype=np.float64)
    z = surface_z.astype(np.float64)
    k = min(3, full_dim - 1)
    if k < 1:
        return z
    spline     = UnivariateSpline(x, z, k=k, s=s, ext='extrapolate')
    smoothed_z = spline(x)
    return smoothed_z


# ─── 2D offset map ────────────────────────────────────────────────────────────

def build_full_offset_map(volume, xz_anchors, yz_anchors,
                          half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias,
                          offset_smooth_sig=OFFSET_SMOOTH_SIG):
    """
    Build the full 2-D offset map by running DP surface detection on every
    individual XZ slice (constant-Y cross-section) of the volume.

    Strategy
    --------
    The user provided anchors on two centre slices:
      • xz_anchors : clicked on  volume[dim_y//2, ...]     (XZ plane, varies with x)
      • yz_anchors : clicked on  volume[..., dim_x//2].T   (YZ plane, varies with y)

    To guide the DP search window for each XZ slice at position y, we need a
    per-column Z hint that accounts for both the X-variation (from xz_anchors)
    and the Y-variation (from yz_anchors).  We compute this as:

        z_hint(y, x) = z_xz_centre(x)  +  [z_yz_centre(y) - z_yz_centre(y_centre)]

    where z_xz_centre(x) is the anchor-interpolated Z from the centre XZ slice,
    and the bracketed term shifts the whole search window up/down according to
    how much the surface drifts along Y.  This gives every slice its own set of
    per-column anchors that correctly track the 3D surface.

    Each XZ slice then runs a full DP with these adapted anchors, producing one
    row of the 2-D offset map from real image data — not from extrapolation.

    Returns
    -------
    offSetMap   : int16 array, shape (dim_y, dim_x)
    z_xz_centre : float array, shape (dim_x,)   — centre-slice surface (for display)
    z_yz_centre : float array, shape (dim_y,)   — centre YZ surface    (for display)
    """
    dim_y, dim_z, dim_x = volume.shape[:3]
    y_centre = dim_y // 2
    x_centre = dim_x // 2

    # ── Step 1: detect surface on the two centre slices (same as preview) ──
    print("  Detecting centre XZ surface curve...")
    xz_slice_c      = volume[y_centre, ...]                  # (dim_z, dim_x)
    z_xz_raw, _     = dp_surface_detection(
        xz_slice_c, xz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias
    )
    z_xz_centre     = smooth_surface_spline(z_xz_raw, dim_x) # (dim_x,)

    print("  Detecting centre YZ surface curve...")
    yz_slice_c      = volume[..., x_centre].T                 # (dim_z, dim_y)
    z_yz_raw, _     = dp_surface_detection(
        yz_slice_c, yz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias
    )
    z_yz_centre     = smooth_surface_spline(z_yz_raw, dim_y)  # (dim_y,)

    # Y-drift relative to the centre row (scalar per y)
    y_drift = z_yz_centre - z_yz_centre[y_centre]            # (dim_y,)

    # ── Step 2: for every XZ slice, build adapted anchors and run DP ──
    # Adapted anchors for slice y:  z_click_adapted = z_xz_centre(x) + y_drift(y)
    # We pass these as a synthetic anchor list covering the full X range so that
    # the DP search window is correctly centred at every column.
    #
    # To keep DP fast we sample N_ADAPTED_ANCHORS evenly spaced columns as anchors.
    N_ADAPTED_ANCHORS = max(N_CLICKS, min(20, dim_x // 10))
    anchor_x_positions = np.round(
        np.linspace(0, dim_x - 1, N_ADAPTED_ANCHORS)
    ).astype(int)

    offSetMap = np.zeros((dim_y, dim_x), dtype=np.float32)

    print(f"  Running per-slice DP on all {dim_y} XZ slices...")
    for y in range(dim_y):
        if y % max(1, dim_y // 20) == 0:
            print(f"    slice {y}/{dim_y}", end='\r')

        # Adapted Z hint for this y: centre curve shifted by y_drift[y]
        z_hint_y = z_xz_centre + y_drift[y]                  # (dim_x,)

        # Build synthetic anchor list from sampled positions
        adapted_anchors = [
            (int(ax), int(np.clip(round(z_hint_y[ax]), 0, dim_z - 1)))
            for ax in anchor_x_positions
        ]

        xz_slice_y   = volume[y, ...]                         # (dim_z, dim_x)
        z_surf_y, _  = dp_surface_detection(
            xz_slice_y, adapted_anchors,
            half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias
        )
        z_surf_smooth        = smooth_surface_spline(z_surf_y, dim_x)
        offSetMap[y, :]      = z_surf_smooth

    print(f"    slice {dim_y}/{dim_y}  done.          ")

    # ── Step 3: convert absolute surface Z positions to shift offsets ──
    # For each column x, the offset at row y is how much the surface at (y,x)
    # deviates from the surface at the centre row (y_centre, x).
    # Using a per-column reference is essential: a scalar mean would introduce
    # X-direction distortion in every non-centre row.
    #
    #   offset(y, x) = -(surface_z(y, x) - surface_z(y_centre, x))
    #
    # This makes the centre row exactly zero (no shift) and every other row
    # shifts relative to the true surface shape at that column.
    reference_z_per_col = offSetMap[y_centre, :]                      # shape (dim_x,)
    offSetMap_raw = -(offSetMap - reference_z_per_col[np.newaxis, :])  # (dim_y, dim_x), float32

    # ── Step 4: 2-D Gaussian smoothing of the offset map ──
    # Per-slice DP produces smooth curves along X but can have row-to-row
    # discontinuities along Y, causing horizontal stripe artefacts in XY planes.
    # A 2-D Gaussian blur (sigma applied equally in both axes) removes these
    # stripes while preserving the large-scale surface shape.
    offSetMap_smooth = gaussian_filter(offSetMap_raw, sigma=offset_smooth_sig)
    offSetMap        = np.trunc(offSetMap_smooth).astype(np.int16)

    return offSetMap, z_xz_centre, z_yz_centre


# ─── Interactive anchor picker ────────────────────────────────────────────────

class AnchorPicker:
    def __init__(self, n_clicks=N_CLICKS):
        self.n_clicks    = n_clicks
        self.coords      = []
        self.click_count = 0
        self.done        = False

    def reset(self):
        self.coords      = []
        self.click_count = 0
        self.done        = False

    def on_press(self, event):
        if self.done:
            return
        if event.button is MouseButton.LEFT:
            if event.xdata is None or event.ydata is None:
                print("  Click inside the image frame.")
                return
            pt = (int(event.xdata), int(event.ydata))
            self.coords.append(pt)
            self.click_count += 1
            print(f"  Anchor {self.click_count}/{self.n_clicks}: x={pt[0]}, z={pt[1]}")
            if self.click_count >= self.n_clicks:
                self.done = True
        elif event.button is MouseButton.RIGHT:
            if self.coords:
                removed = self.coords.pop()
                self.click_count -= 1
                print(f"  Removed {removed}; remaining: {self.click_count}/{self.n_clicks}")
            else:
                print("  No anchors to remove.")

    def collect(self, ax, fig, image, title):
        self.reset()
        ax.clear()
        ax.imshow(image, cmap='gray', aspect='auto')
        ax.set_title(title, fontsize=9)
        fig.canvas.draw()
        cid = fig.canvas.mpl_connect('button_press_event', self.on_press)
        while not self.done:
            plt.pause(0.3)
        fig.canvas.mpl_disconnect(cid)
        print(f"  Anchors: {self.coords}")
        return list(self.coords)


# ─── Preview / confirm loop ───────────────────────────────────────────────────

def run_preview(volume, xz_anchors, yz_anchors,
                half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias,
                fig, axes):
    """
    Detect surface, apply to center slices, display result.
    Returns (confirmed: bool, offSetMap | None).
    """
    dim_y, dim_z, dim_x = volume.shape[:3]
    ax_a, ax_b, ax_c, ax_d = axes

    print(f"\n  Running DP detection (half_win={half_win}, λ={dp_lambda}, "
          f"tol={anchor_tol}, bias={anchor_bias})...")

    # ── XZ detection & preview ──
    xz_slice      = volume[dim_y // 2, ...]
    z_xz, cm_xz  = dp_surface_detection(
        xz_slice, xz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias
    )
    z_xz_s        = smooth_surface_spline(z_xz, dim_x)

    # ── YZ detection & preview ──
    yz_slice      = volume[..., dim_x // 2].T
    z_yz, cm_yz  = dp_surface_detection(
        yz_slice, yz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol, anchor_bias
    )
    z_yz_s        = smooth_surface_spline(z_yz, dim_y)

    # ── Build preview offset map (separable) ──
    xMap2d = np.tile(z_xz_s - z_xz_s[0], (dim_y, 1))
    yMap2d = np.tile((z_yz_s - z_yz_s[0]), (dim_x, 1)).T
    offSetMap_preview = np.trunc(xMap2d + yMap2d).astype(np.int16) * -1

    # Apply only to centre slices
    def preview_roll_slice_xz(sl, offset_row):
        """sl: (dim_z, dim_x), offset_row: (dim_x,) → rolled (dim_z, dim_x)"""
        vol1 = sl[np.newaxis, :, :]       # (1, dim_z, dim_x)
        off1 = offset_row[np.newaxis, :]  # (1, dim_x)
        return fast_roll_along_z(vol1, off1)[0]

    def preview_roll_slice_yz(sl, offset_col):
        """sl: (dim_z, dim_y) [transposed YZ], offset_col: (dim_y,) → rolled (dim_z, dim_y)"""
        # Treat dim_y as the "X" axis for fast_roll_along_z by passing shape (1, dim_z, dim_y)
        vol1 = sl[np.newaxis, :, :]        # (1, dim_z, dim_y)
        off1 = offset_col[np.newaxis, :]   # (1, dim_y)
        return fast_roll_along_z(vol1, off1)[0]

    # XZ centre offset: row y_centre of preview map → shape (dim_x,)
    rolled_xz = preview_roll_slice_xz(xz_slice, offSetMap_preview[dim_y // 2, :])
    # YZ centre offset: column x_centre of preview map → shape (dim_y,)
    rolled_yz = preview_roll_slice_yz(yz_slice,  offSetMap_preview[:, dim_x // 2])

    # ── Plots ──
    x_axis = np.arange(dim_x)
    y_axis = np.arange(dim_y)

    ax_a.clear(); ax_a.imshow(xz_slice, cmap='gray', aspect='auto')
    ax_a.plot(x_axis, z_xz,   'g-',  lw=0.8, alpha=0.6, label='DP raw')
    ax_a.plot(x_axis, z_xz_s, 'b-',  lw=1.5, label='Spline')
    ax_a.scatter([c[0] for c in xz_anchors], [c[1] for c in xz_anchors],
                 color='red', s=50, zorder=5, label='Anchors')
    ax_a.set_title(f"XZ detected surface  (half_win={half_win})", fontsize=9)
    ax_a.legend(fontsize=7, loc='upper right')

    ax_b.clear(); ax_b.imshow(yz_slice, cmap='gray', aspect='auto')
    ax_b.plot(y_axis, z_yz,   'g-',  lw=0.8, alpha=0.6, label='DP raw')
    ax_b.plot(y_axis, z_yz_s, 'b-',  lw=1.5, label='Spline')
    ax_b.scatter([c[0] for c in yz_anchors], [c[1] for c in yz_anchors],
                 color='red', s=50, zorder=5, label='Anchors')
    ax_b.set_title(f"YZ detected surface  (half_win={half_win})", fontsize=9)
    ax_b.legend(fontsize=7, loc='upper right')

    ax_c.clear(); ax_c.imshow(rolled_xz, cmap='gray', aspect='auto')
    ax_c.set_title("XZ — flattened preview", fontsize=9)

    ax_d.clear(); ax_d.imshow(rolled_yz, cmap='gray', aspect='auto')
    ax_d.set_title("YZ — flattened preview", fontsize=9)

    fig.canvas.draw()
    plt.pause(0.1)

    confirmed = messagebox.askyesno(
        "Confirm registration?",
        f"Parameters:  half_win={half_win}  λ={dp_lambda}  "
        f"anchor_tol={anchor_tol}  anchor_bias={anchor_bias}\n\n"
        "Does the flattened preview look correct?\n\n"
        "  Yes → proceed to full 3D registration\n"
        "  No  → adjust parameters and retry"
    )
    return confirmed, offSetMap_preview if confirmed else None


# ─── Main ─────────────────────────────────────────────────────────────────────

def ask_params(current_half_win, current_lambda, current_tol, current_bias, current_smooth2d):
    """Pop up a dialog asking for updated parameters."""
    tk2 = Tk(); tk2.withdraw(); tk2.attributes("-topmost", True)

    prompt = (
        f"Current parameters:\n"
        f"  half_win       = {current_half_win}   (local Z search window, px)\n"
        f"  dp_lambda      = {current_lambda}   (smoothness penalty)\n"
        f"  anchor_tol     = {current_tol}   (hard constraint radius, px)\n"
        f"  anchor_bias    = {current_bias}   (distance-to-anchor bias weight)\n"
        f"  offset_smooth  = {current_smooth2d}   (2-D Gaussian sigma for offset map, px)\n\n"
        "Enter new values as:  half_win  lambda  anchor_tol  anchor_bias  offset_smooth\n"
        "Example:  7  10  10  0.3  5\n"
        "  anchor_bias: 0=no bias, 0.1~0.5=soft pull, >1.0=strong pull\n"
        "  offset_smooth: 2D Gaussian sigma; 0=off, 3~15 recommended\n"
        "(press Cancel to exit without registration)"
    )
    raw = simpledialog.askstring("Adjust parameters", prompt, parent=tk2)
    tk2.destroy()

    if raw is None:
        return None
    parts = raw.strip().split()
    try:
        hw     = max(2,   int(parts[0]))
        lam    = max(0.0, float(parts[1]))
        tol    = max(2,   int(parts[2]))
        bias   = max(0.0, float(parts[3])) if len(parts) > 3 else current_bias
        sig2d  = max(0.0, float(parts[4])) if len(parts) > 4 else current_smooth2d
        return hw, lam, tol, bias, sig2d
    except (ValueError, IndexError):
        print("  Could not parse input; keeping current values.")
    return current_half_win, current_lambda, current_tol, current_bias, current_smooth2d


def main():
    # ── File selection ──
    tk = Tk(); tk.withdraw(); tk.attributes("-topmost", True)
    stack_file_path = filedialog.askopenfilename(
        title="Select the .bin index file",
        filetypes=[("", "*.bin")]
    )
    tk.destroy()
    if not stack_file_path:
        print("No file selected. Exiting.")
        return

    DataId        = os.path.basename(stack_file_path)
    root          = os.path.dirname(stack_file_path)
    string_DataId = DataId[:-4]

    tif_path        = root + '/' + string_DataId + '_3d_view.tif'
    offset_map_path = os.path.join(root, 'offset_map.npy')

    # ══════════════════════════════════════════════════════════════════════════
    if Is_RegistUsingOffsetMap:
        # ── Fast path: load existing offset map and apply directly ──
        if not os.path.exists(offset_map_path):
            print(f"ERROR: offset_map.npy not found at {offset_map_path}")
            print("  Set Is_RegistUsingOffsetMap = False to run the full detection pipeline.")
            return
        offSetMap_full = np.load(offset_map_path)
        print(f"Loaded offset map: {offset_map_path}  shape={offSetMap_full.shape}")

        print(f"\nApplying registration to: {tif_path}")
        volume_registor(tif_path, offSetMap_full)

        # Uncomment additional volumes as needed:
        # try:
        #     volume_registor(root + '/' + string_DataId + '_IntImg_meanFreq.tif', offSetMap_full)
        # except FileNotFoundError:
        #     print("  meanFreq not found — skipped.")
        # try:
        #     volume_registor(root + '/' + string_DataId + '_IntImg_LIV_raw.tif', offSetMap_full)
        #     volume_registor(root + '/' + string_DataId + '_IntImg_LIV.tif',     offSetMap_full)
        # except FileNotFoundError:
        #     print("  LIV not found — skipped.")
        # try:
        #     volume_registor(root + '/' + string_DataId + '_IntImg_mLIV_raw.tif', offSetMap_full)
        #     volume_registor(root + '/' + string_DataId + '_IntImg_mLIV.tif',     offSetMap_full)
        # except FileNotFoundError:
        #     print("  mLIV not found — skipped.")
        # try:
        #     volume_registor(root + '/' + string_DataId + '_IntImg_aliv.tif',      offSetMap_full)
        #     volume_registor(root + '/' + string_DataId + '_IntImg_dbOct.tif',     offSetMap_full)
        #     volume_registor(root + '/' + string_DataId + '_IntImg_swiftness.tif', offSetMap_full)
        #     for pat in ['_IntImg_aliv_min*-max*.tif', '_IntImg_swiftness_min*-max*.tif']:
        #         hits = glob.glob(root + '/' + string_DataId + pat)
        #         if hits: volume_registor(hits[0], offSetMap_full)
        # except FileNotFoundError:
        #     print("  aLiv/swiftness not found — skipped.")

        print("\nAll done.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # ── Full pipeline: manual picking → DP detection → confirm → register ──
    print(f"Loading: {tif_path}")
    raw_data = load_full_tiffstack(tif_path)
    dim_y, dim_z, dim_x = raw_data.shape[:3]
    print(f"Volume shape: Y={dim_y}, Z={dim_z}, X={dim_x}")

    # ── Display setup ──
    fig, axes_all = plt.subplots(2, 2, figsize=(8, 15), num=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02, hspace=0.08, wspace=0.05)
    axes = (axes_all[0, 0], axes_all[0, 1], axes_all[1, 0], axes_all[1, 1])
    plt.pause(0.1)

    picker = AnchorPicker(n_clicks=N_CLICKS)

    # ── Collect XZ anchors ──
    print(f"\n=== XZ plane: click {N_CLICKS} points ON the surface ===")
    print("  Left-click = add,  Right-click = undo")
    print("  TIP: click directly on the surface edge you want to flatten.")
    xz_anchors = picker.collect(
        axes[0], fig,
        raw_data[dim_y // 2, ...],
        f"XZ click {N_CLICKS} surface (left=add, right=undo)"
    )

    # ── Collect YZ anchors ──
    print(f"\n=== YZ click {N_CLICKS} points ON the surface ===")
    yz_anchors = picker.collect(
        axes[1], fig,
        raw_data[..., dim_x // 2].T,
        f"YZ click {N_CLICKS} surface (left=add, right=undo)"
    )

    # ── Preview / confirm loop ──
    half_win          = HALF_WIN
    dp_lambda         = DP_LAMBDA
    anchor_tol        = ANCHOR_TOL
    anchor_bias       = ANCHOR_BIAS
    offset_smooth_sig = OFFSET_SMOOTH_SIG
    confirmed         = False

    while not confirmed:
        confirmed, _ = run_preview(
            raw_data, xz_anchors, yz_anchors,
            half_win, smooth_sig=SMOOTH_SIG,
            dp_lambda=dp_lambda, anchor_tol=anchor_tol, anchor_bias=anchor_bias,
            fig=fig, axes=axes
        )
        if not confirmed:
            result = ask_params(half_win, dp_lambda, anchor_tol, anchor_bias, offset_smooth_sig)
            if result is None:
                print("\nUser cancelled. Exiting without registration.")
                plt.close('all')
                return
            half_win, dp_lambda, anchor_tol, anchor_bias, offset_smooth_sig = result

    # ── Build full 2D offset map ──
    print("\nBuilding full 2D offset map...")
    offSetMap_full, _, _ = build_full_offset_map(
        raw_data, xz_anchors, yz_anchors,
        half_win, SMOOTH_SIG, dp_lambda, anchor_tol, anchor_bias, offset_smooth_sig
    )

    # ── Save offset map ──
    np.save(offset_map_path, offSetMap_full)
    print(f"  Offset map saved: {offset_map_path}  shape={offSetMap_full.shape}")

    del raw_data

    # ── Apply registration ──
    print("\nApplying registration...")
    volume_registor(tif_path, offSetMap_full)

    # Uncomment additional volumes as needed:
    # try:
    #     volume_registor(root + '/' + string_DataId + '_IntImg_meanFreq.tif', offSetMap_full)
    # except FileNotFoundError:
    #     print("  meanFreq not found — skipped.")
    # try:
    #     volume_registor(root + '/' + string_DataId + '_IntImg_LIV_raw.tif', offSetMap_full)
    #     volume_registor(root + '/' + string_DataId + '_IntImg_LIV.tif',     offSetMap_full)
    # except FileNotFoundError:
    #     print("  LIV not found — skipped.")
    # try:
    #     volume_registor(root + '/' + string_DataId + '_IntImg_mLIV_raw.tif', offSetMap_full)
    #     volume_registor(root + '/' + string_DataId + '_IntImg_mLIV.tif',     offSetMap_full)
    # except FileNotFoundError:
    #     print("  mLIV not found — skipped.")
    # try:
    #     volume_registor(root + '/' + string_DataId + '_IntImg_aliv.tif',      offSetMap_full)
    #     volume_registor(root + '/' + string_DataId + '_IntImg_dbOct.tif',     offSetMap_full)
    #     volume_registor(root + '/' + string_DataId + '_IntImg_swiftness.tif', offSetMap_full)
    #     for pat in ['_IntImg_aliv_min*-max*.tif', '_IntImg_swiftness_min*-max*.tif']:
    #         hits = glob.glob(root + '/' + string_DataId + pat)
    #         if hits: volume_registor(hits[0], offSetMap_full)
    # except FileNotFoundError:
    #     print("  aLiv/swiftness not found — skipped.")

    print("\nAll done.")
    plt.show()


if __name__ == "__main__":
    main()