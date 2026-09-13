"""
Verifies: the arc-length integration method in extract_arc_length_segment()
(gatekeeper_v5.py): S = cumulative sum of consecutive Euclidean step
lengths between sampled (X, Y) points.

This is polygonal (chord-length) approximation of arc length, which is a
provably CONVERGENT, one-sided (always <= true length) estimator for any
rectifiable curve as sample density increases - this is a standard result
in real analysis (arc length is the supremum of inscribed polygon
lengths). We verify this directly against curves with an exact, closed-
form analytic length:
  1. A straight line (trivial: any sampling gives the exact length).
  2. A circular arc of known radius and angle (length = radius * angle).
  3. Convergence rate as a function of sampling density, to confirm the
     approximation error shrinks as expected (not just "looks close").
"""
import numpy as np


def polygon_arc_length(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def straight_line_case():
    x = np.linspace(0, 100, 5)  # very sparse - a straight line needs no density
    y = np.zeros_like(x)
    est = polygon_arc_length(x, y)
    true_len = 100.0
    print(f"Straight line: true=100.0000  estimate={est:.4f}  "
          f"error={abs(est-true_len):.2e}  [{'OK' if abs(est-true_len) < 1e-9 else 'FAIL'}]")


def circular_arc_case(radius=50.0, angle_deg=90.0, n_points_list=(4, 8, 16, 32, 64, 256, 2048)):
    true_len = radius * np.deg2rad(angle_deg)
    print(f"\nCircular arc: radius={radius}, angle={angle_deg} deg, true length={true_len:.6f}")
    print(f"{'n_points':>10}{'estimate':>14}{'abs error':>14}{'error ratio to prev':>22}")
    prev_err = None
    for n in n_points_list:
        theta = np.linspace(0, np.deg2rad(angle_deg), n)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        est = polygon_arc_length(x, y)
        err = true_len - est  # polygon length is always <= true arc length
        ratio = (prev_err / err) if (prev_err is not None and err != 0) else float('nan')
        print(f"{n:>10}{est:>14.6f}{err:>14.2e}{ratio:>22.2f}")
        prev_err = err
    print("Expectation: estimate is always <= true length (chords are shorter than the arc\n"
          "they subtend), error shrinks monotonically, and roughly quadruples the accuracy\n"
          "(error ratio ~4x) each time n_points doubles - consistent with the O(1/n^2)\n"
          "discretization error bound for smooth curves. This confirms convergence, not\n"
          "just a single lucky match.")


def resampling_density_invariance_case(radius=1000.0, angle_deg=90.0, grid_step=1.0):
    """
    The gatekeeper interpolates all laps onto a common S-grid with a fixed
    step size (1.0 telemetry unit in the original code). Verify that
    interpolating a densely-sampled curve onto that grid and re-measuring
    its arc length matches the KNOWN analytic chord-approximation error
    bound, not just "looks close":

        error_per_segment ~= R * (d_theta)^2 / 8   (small-angle expansion
                                                      of R - R*cos(d_theta/2))
        total_error        ~= R * theta_total^2 / (8 * n_segments)

    radius=1000 units is representative of typical telemetry-unit corner
    radii once a units-per-metre calibration factor (e.g. ~10 units/m,
    corresponding to a ~100m real corner radius) is applied - i.e. the
    scale at which the corridor's 1.0-unit grid step is actually used in
    practice.
    """
    theta_dense = np.linspace(0, np.deg2rad(angle_deg), 5000)
    x_dense = radius * np.cos(theta_dense)
    y_dense = radius * np.sin(theta_dense)
    s_dense = np.concatenate(([0.0], np.cumsum(np.sqrt(np.diff(x_dense) ** 2 + np.diff(y_dense) ** 2))))

    common_s = np.arange(0, s_dense[-1], grid_step)
    x_interp = np.interp(common_s, s_dense, x_dense)
    y_interp = np.interp(common_s, s_dense, y_dense)
    reprojected_len = polygon_arc_length(x_interp, y_interp)
    true_len = radius * np.deg2rad(angle_deg)
    observed_err = true_len - reprojected_len

    # Two DISTINCT error sources, isolated separately:
    #  (a) truncation: np.arange(0, stop, step) never emits a point at
    #      `stop` itself, so up to one grid_step of arc length at the far
    #      end is silently dropped from the projected corridor.
    #  (b) curvature discretization: exact chord-vs-arc formula for n
    #      equal-angle partitions of a circular arc,
    #      error = R*theta - 2*R*n*sin(theta/(2n))
    missing_tail = s_dense[-1] - common_s[-1]
    n_segments = len(common_s) - 1
    theta_total = np.deg2rad(angle_deg)
    curvature_err = radius * theta_total - 2 * radius * n_segments * np.sin(theta_total / (2 * n_segments))

    print(f"\nResampling-onto-common-grid check (radius={radius}, grid_step={grid_step}):")
    print(f"  true length              = {true_len:.4f}")
    print(f"  reprojected length       = {reprojected_len:.4f}")
    print(f"  observed error           = {observed_err:.4f}")
    print(f"  (a) truncated tail length (np.arange boundary artifact) = {missing_tail:.4f}")
    print(f"  (b) curvature discretization error (exact chord formula) = {curvature_err:.6f}")
    print(f"  (a)+(b)                  = {missing_tail + curvature_err:.4f}  vs observed {observed_err:.4f}")
    combined = missing_tail + curvature_err
    match = abs(combined - observed_err) < 1e-3
    print(f"  [{'OK - fully explained by these two known, quantified effects' if match else 'CHECK'}]")
    print("\n  Practical implication for gatekeeper_v5.py: curvature discretization error (b)\n"
          "  is negligible at this scale (~1e-4 or smaller). The dominant effect is (a), a\n"
          "  pure grid-construction boundary artifact - the LAST UP-TO-ONE grid_step of the\n"
          "  hotspot is silently excluded from every lap's corridor projection. This is not\n"
          "  dangerous (same exclusion applies identically to every lap, reference and test\n"
          "  alike, so it does not bias the infraction bound) but it should be fixed for full\n"
          "  rigor: use np.linspace(0, min_S, num=int(min_S/grid_step)+1) instead of\n"
          "  np.arange, so the grid spans the full corridor including its endpoint.")


if __name__ == "__main__":
    straight_line_case()
    circular_arc_case()
    resampling_density_invariance_case()
