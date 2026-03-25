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
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline

"""
Prompt:
我目前代码实现的功能是通过手动选择物体表面轮廓5点+多项式拟合的方法，将物体的三维图像沿着表面轮廓“拉平”（或者叫registration）。
具体实现方式如下：读取一个三维tiff图，显示两幅过中心的XZ、YZ截面图（对应代码中的`raw_data[round(dim_y/2), ...]`和`raw_data[..., round(dim_x/2)].T`），用户分别在XZ图和YZ图上沿着物体表面手动选择5个点，代码对选择的点各自进行多项式拟合，并插值到3D空间，得到物体表面轮廓曲面，最后沿着这个曲面对整个三维图像进行“拉平”。
目前不足的地方在于，表面轮廓的拉平精度极度依赖用户选择点是否准确、多项式拟合是否符合实际物体表面轮廓，无法做到根据图像中的实际轮廓信息自动计算出轮廓曲线、曲面。我有考虑过引入机器视觉模块来自动判断轮廓曲线，但是实际应用时，由于图像内容的复杂性，我期望拉平的轮廓曲线的周围（上下方向）常有其他伪影存在，且表面轮廓所在的Z轴位置常常不固定，所以我担心无法做到自动识别哪一段Z轴区域内才适用于自动轮廓识别。
目前我的想法是，同样还是显示两幅过中心的XZ、YZ截面图，用户还是可以在两幅图中手动选择若干个（例如5个）表面轮廓的点。但是不再使用多项式拟合这种低精度的方法去估计表面轮廓曲线，而是使用其他更robust的自动表面轮廓识别算法（细节在最后一段进行补充)，只是其适用范围不再是整幅XZ图和YZ图，而是在用户选择点周围的一小段Z轴区域内。计算出显示的XZ、YZ截面图中的表面轮廓曲线后，先尝试对这两幅图进行表面拉平，并更新图像让用户检验拉平效果是否合格（如果不合格的话可以在代码里微调“用户选择点周围的一小段Z轴区域”的区域范围，比如说上下50pixel > 10pixel）。如果用户觉得拉平效果合格、并点击确认后，算法自动估算出整个三维图中的合适的“用户选择点周围的一小段Z轴区域”，并在该范围内提取表面轮廓曲面，最后对整个三维图进行拉平操作。
有一点需要注意的是，数据集中常见的物体表面轮廓，通常并非是一小段范围内的最高强度点，因为物体通常为强散射+低轴向分辨率，意味着物体表面轮廓处的特征更多是强度从低到高的突变，且表面轮廓的边界处通常稍显模糊、不会特别锋利，即对比度不会很理想；同时，也会有其表面轮廓的上下部分同时存在其他强散射的分层结构，但关键是这些“非表面轮廓”的分层结构与目标的表面轮廓一样，有几乎相同的轮廓线（可以理解为只加了不同的Z方向的offset，所以有时候当样品的目标表面轮廓层的对比度不清晰时，用户可以手动选择那些对比度更高的“非表面轮廓”层，来达到同样的三维图拉平效果）。所以我猜测，简单的“平滑+峰值检测”可能无法满足这个情况，请仔细论证、考虑其他更robust的轮廓检测算法，例如cv2之类的）

Surface flattening via CIDI cost map + Dynamic Programming + anchor constraints.

Why this approach:
  The target surface is characterised by a LOW→HIGH intensity transition along Z
  (strong backscattering object, low axial resolution). Multiple parallel
  scattering layers exist above and below the true surface, all with nearly
  identical Z-shape. Simple gradient-peak detection therefore gives ambiguous
  results — every layer boundary is a valid peak.

  This implementation:
    1. Builds a per-column COST MAP that scores each Z position by how well it
       sits at a rising-edge transition (CIDI: Cumulative Intensity Difference
       with Direction). The cost is LOWEST at positions where intensity transitions
       sharply from low to high going deeper into the sample.
    2. Solves for the globally optimal SURFACE PATH across all columns using
       Dynamic Programming (DP), with a smoothness penalty that prevents the
       detected surface from jumping between parallel layers.
    3. ANCHORS the DP search: at columns where the user clicked, the DP is
       constrained to pass within ±ANCHOR_TOL pixels of the clicked Z. This
       resolves the layer-selection ambiguity — the user's click says "this layer,
       not the one 50 px above".

Tunable parameters (top of file):
  HALF_WIN     : local Z search window half-width around each anchor (px). Default 60.
  SMOOTH_SIG   : Gaussian sigma for Z-profile smoothing before CIDI.       Default 3.
  DP_LAMBDA    : DP smoothness penalty weight λ·|Δz|².                     Default 5.
  ANCHOR_TOL   : Hard constraint radius around each anchor click (px).     Default 15.
  SPLINE_S     : UnivariateSpline smoothing factor (None = auto).
  N_CLICKS     : Anchor clicks per plane.                                   Default 5.
"""

# ─── Tunable parameters ───────────────────────────────────────────────────────
HALF_WIN   = 10     # Local Z search window = clicked_z ± HALF_WIN
SMOOTH_SIG = 1.0    # Gaussian sigma for Z-profile smoothing
DP_LAMBDA  = 10.0    # DP smoothness penalty: λ * |Δz|^2 per adjacent column. 控制路径的平滑程度，如果检测到的曲线在列间跳动，应当增大
ANCHOR_TOL = 5    # Hard DP constraint radius around each anchor (px)。 控制锚点约束的松紧，如果用户点击精度有限可以适当放大，但如果层间距很小则需要收紧。
SPLINE_S   = None   # UnivariateSpline smoothing (None = auto, 0 = exact)
N_CLICKS   = 5      # Anchor clicks per plane
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

def cidi_cost_column(profile_1d, z_lo, z_hi, smooth_sig):
    """
    Compute a per-Z cost vector for one column profile within [z_lo, z_hi].

    CIDI logic:
      For each candidate surface Z = s, we want the intensity to be LOW above s
      and HIGH below s (i.e. a rising edge when going from outside to inside).

      cost(s) = - [ mean(I[s:s+w]) - mean(I[s-w:s]) ]
              = - (mean of w pixels below s)  +  (mean of w pixels above s)

      This is MINIMISED at positions where the intensity jumps from low to high.
      w is chosen adaptively as half the local window width.

    Returns:
      cost : float array, length = z_hi - z_lo + 1  (local indices)
    """
    dim_z   = len(profile_1d)
    z_lo    = max(0, z_lo)
    z_hi    = min(dim_z - 1, z_hi)
    win_len = z_hi - z_lo + 1

    if win_len < 4:
        return np.zeros(win_len)

    local   = gaussian_filter1d(
        profile_1d[z_lo: z_hi + 1].astype(np.float32), sigma=smooth_sig
    )
    # half-window for the above/below averaging
    w       = max(2, win_len // 6)
    cost    = np.zeros(win_len, dtype=np.float32)

    for s in range(win_len):
        # pixels BELOW s (inside the object): should be high
        lo_start = max(0, s - w)
        below    = local[lo_start: s].mean() if s > lo_start else local[s]
        # pixels ABOVE s (outside the object): should be low
        hi_end   = min(win_len, s + w)
        above    = local[s: hi_end].mean() if hi_end > s else local[s]
        # cost is low where below is high and above is low
        cost[s]  = above - below

    return cost.astype(np.float64)


# ─── DP surface detection ─────────────────────────────────────────────────────

def dp_surface_detection(slice_2d, anchors, half_win, smooth_sig,
                          dp_lambda, anchor_tol):
    """
    Detect the surface curve in a 2-D slice (shape: [dim_z, dim_x]) using
    CIDI cost map + Dynamic Programming + anchor hard constraints.

    Parameters
    ----------
    slice_2d   : 2-D array, shape (dim_z, dim_x)
    anchors    : list of (x_click, z_click) tuples from user
    half_win   : local Z-search half-width (px)
    smooth_sig : Gaussian sigma for CIDI
    dp_lambda  : smoothness penalty weight
    anchor_tol : radius of hard constraint at anchor columns

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

    # ── 2. Interpolate z_lo / z_hi for every column from anchors ──
    #    Between anchors: linearly interpolate the search window centre.
    #    Outside the leftmost/rightmost anchor: extrapolate (clamp).
    if not anchor_dict:
        raise ValueError("No anchors provided.")

    ax_sorted  = sorted(anchor_dict.keys())
    z_centres  = np.array([anchor_dict[x][0] for x in ax_sorted], dtype=np.float64)
    ax_arr     = np.array(ax_sorted, dtype=np.float64)
    all_x      = np.arange(dim_x, dtype=np.float64)
    z_c_interp = np.interp(all_x, ax_arr, z_centres)           # clamps outside
    z_lo_all   = np.clip(z_c_interp - half_win, 0, dim_z - 1).astype(int)
    z_hi_all   = np.clip(z_c_interp + half_win, 0, dim_z - 1).astype(int)

    # ── 3. Build full cost map (dim_z, dim_x) — NaN outside window ──
    cost_map = np.full((dim_z, dim_x), np.nan, dtype=np.float64)
    for x in range(dim_x):
        zl, zh = z_lo_all[x], z_hi_all[x]
        col_cost = cidi_cost_column(slice_2d[:, x], zl, zh, smooth_sig)
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
                           half_win, smooth_sig, dp_lambda, anchor_tol):
    """
    Detect the surface across the entire volume and build the 2-D offset map.

    Strategy:
      • For each Y row (constant Y), run DP on the XZ slice → surface_z(x | y)
        But computing DP for every Y is slow. Instead:
        - Run DP on the centre XZ slice → get surface_z_xz(x)   [1-D, varies with x]
        - Run DP on the centre YZ slice → get surface_z_yz(y)   [1-D, varies with y]
        - Combine additively (same as original code):
            offset(y, x) = -(Δz_x(x) + Δz_y(y))
          where Δz_x(x) = surface_z_xz(x) - surface_z_xz(0)
                Δz_y(y) = surface_z_yz(y) - surface_z_yz(0)
      
      This is the same separable approximation as the original code, now with
      a far more accurate 1-D curve estimate from the DP detector.
    """
    dim_y, dim_z, dim_x = volume.shape[:3]

    print("  Detecting XZ surface curve...")
    xz_slice               = volume[dim_y // 2, ...]        # (dim_z, dim_x)
    z_xz, _                = dp_surface_detection(
        xz_slice, xz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol
    )
    z_xz_smooth            = smooth_surface_spline(z_xz, dim_x)

    print("  Detecting YZ surface curve...")
    yz_slice               = volume[..., dim_x // 2].T       # (dim_z, dim_y)
    z_yz, _                = dp_surface_detection(
        yz_slice, yz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol
    )
    z_yz_smooth            = smooth_surface_spline(z_yz, dim_y)

    # Separable combination
    xMap2d = np.tile(z_xz_smooth - z_xz_smooth[0], (dim_y, 1))       # (dim_y, dim_x)
    yMap2d = np.tile((z_yz_smooth - z_yz_smooth[0]), (dim_x, 1)).T    # (dim_y, dim_x)
    offSetMap = np.trunc(xMap2d + yMap2d).astype(np.int16) * -1

    return offSetMap, z_xz_smooth, z_yz_smooth


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
        ax.imshow(image, cmap='gray')
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
                half_win, smooth_sig, dp_lambda, anchor_tol,
                fig, axes):
    """
    Detect surface, apply to center slices, display result.
    Returns (confirmed: bool, offSetMap | None).
    """
    dim_y, dim_z, dim_x = volume.shape[:3]
    ax_a, ax_b, ax_c, ax_d = axes

    print(f"\n  Running DP detection (half_win={half_win}, λ={dp_lambda}, tol={anchor_tol})...")

    # ── XZ detection & preview ──
    xz_slice      = volume[dim_y // 2, ...]
    z_xz, cm_xz  = dp_surface_detection(
        xz_slice, xz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol
    )
    z_xz_s        = smooth_surface_spline(z_xz, dim_x)

    # ── YZ detection & preview ──
    yz_slice      = volume[..., dim_x // 2].T
    z_yz, cm_yz  = dp_surface_detection(
        yz_slice, yz_anchors, half_win, smooth_sig, dp_lambda, anchor_tol
    )
    z_yz_s        = smooth_surface_spline(z_yz, dim_y)

    # ── Build preview offset map (separable) ──
    xMap2d = np.tile(z_xz_s - z_xz_s[0], (dim_y, 1))
    yMap2d = np.tile((z_yz_s - z_yz_s[0]), (dim_x, 1)).T
    offSetMap_preview = np.trunc(xMap2d + yMap2d).astype(np.int16) * -1

    # Apply only to centre slices
    def preview_roll_slice(sl, offset_row):
        """sl: (dim_z, dim_x), offset_row: (dim_x,)"""
        vol1 = sl[np.newaxis, :, :]
        off1 = offset_row[np.newaxis, :]
        return fast_roll_along_z(vol1, off1)[0]

    rolled_xz = preview_roll_slice(xz_slice, offSetMap_preview[dim_y // 2, :])
    rolled_yz = preview_roll_slice(yz_slice,  offSetMap_preview[:, dim_x // 2])

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

    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.1)

    confirmed = messagebox.askyesno(
        "Confirm registration?",
        f"Parameters:  half_win={half_win}  λ={dp_lambda}  anchor_tol={anchor_tol}\n\n"
        "Does the flattened preview look correct?\n\n"
        "  Yes → proceed to full 3D registration\n"
        "  No  → adjust parameters and retry"
    )
    return confirmed, offSetMap_preview if confirmed else None


# ─── Main ─────────────────────────────────────────────────────────────────────

def ask_params(current_half_win, current_lambda, current_tol):
    """Pop up a dialog asking for updated parameters."""
    tk2 = Tk(); tk2.withdraw(); tk2.attributes("-topmost", True)

    prompt = (
        f"Current parameters:\n"
        f"  half_win   = {current_half_win}  (local Z search window, px)\n"
        f"  dp_lambda  = {current_lambda}   (smoothness penalty)\n"
        f"  anchor_tol = {current_tol}  (hard constraint radius, px)\n\n"
        "Enter new values as:  half_win  lambda  anchor_tol\n"
        "Example:  40  5  10\n"
        "(press Cancel to keep current values)"
    )
    raw = simpledialog.askstring("Adjust parameters", prompt, parent=tk2)
    tk2.destroy()

    if raw:
        parts = raw.strip().split()
        try:
            hw  = max(2,   int(parts[0]))
            lam = max(0.0, float(parts[1]))
            tol = max(2,   int(parts[2]))
            return hw, lam, tol
        except (ValueError, IndexError):
            print("  Could not parse input; keeping current values.")
    return current_half_win, current_lambda, current_tol


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

    tif_path = root + '/' + string_DataId + '_3d_view.tif'
    print(f"Loading: {tif_path}")
    raw_data = load_full_tiffstack(tif_path)
    dim_y, dim_z, dim_x = raw_data.shape[:3]
    print(f"Volume shape: Y={dim_y}, Z={dim_z}, X={dim_x}")

    # ── Display setup ──
    fig, axes_all = plt.subplots(2, 2, figsize=(7, 12), num=10)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle("Surface Registration — CIDI + DP + Anchor Constraints", fontsize=11)
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
    half_win  = HALF_WIN
    dp_lambda = DP_LAMBDA
    anchor_tol = ANCHOR_TOL
    confirmed  = False

    while not confirmed:
        confirmed, _ = run_preview(
            raw_data, xz_anchors, yz_anchors,
            half_win, smooth_sig=SMOOTH_SIG,
            dp_lambda=dp_lambda, anchor_tol=anchor_tol,
            fig=fig, axes=axes
        )
        if not confirmed:
            half_win, dp_lambda, anchor_tol = ask_params(half_win, dp_lambda, anchor_tol)

    # ── Build full 2D offset map ──
    print("\nBuilding full 2D offset map...")
    offSetMap_full, _, _ = build_full_offset_map(
        raw_data, xz_anchors, yz_anchors,
        half_win, SMOOTH_SIG, dp_lambda, anchor_tol
    )
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
