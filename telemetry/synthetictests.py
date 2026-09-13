import numpy as np
import sys
sys.path.insert(0, '.')
from boundary_calibrator import project_to_frenet, build_dense_centerline

def make_straight_centerline():
    """Straight line along x-axis, y=0, from x=0 to x=100."""
    s = np.arange(0, 100, 0.1)
    x = s.copy()
    y = np.zeros_like(s)
    return {'S': s, 'X': x, 'Y': y, 'dX': np.gradient(x), 'dY': np.gradient(y)}

def make_arc_centerline(radius=50.0, angle_span=np.pi/2):
    """Quarter-circle arc, counter-clockwise, centered at origin."""
    theta = np.linspace(0, angle_span, 500)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    s = radius * theta
    return {'S': s, 'X': x, 'Y': y, 'dX': np.gradient(x), 'dY': np.gradient(y)}

def approx(a, b, tol=0.05):
    return abs(a - b) <= tol

def test_straight_line_offsets():
    cl = make_straight_centerline()
    # Points at x=50, offset above (y=+2) and below (y=-2) the line
    x = np.array([50.0, 50.0, 50.0])
    y = np.array([0.0, 2.0, -2.0])
    offsets = project_to_frenet(x, y, cl)
    print(f"Straight line test -> offsets: {offsets}")
    assert approx(offsets[0], 0.0), "On-centerline point should have ~0 offset"
    assert approx(abs(offsets[1]), 2.0), "Should measure 2.0m perpendicular distance"
    assert approx(abs(offsets[2]), 2.0), "Should measure 2.0m perpendicular distance"
    assert np.sign(offsets[1]) != np.sign(offsets[2]), "Opposite sides must have opposite signs"
    print("PASS: straight_line_offsets\n")

def test_arc_offsets():
    cl = make_arc_centerline(radius=50.0)
    # A point radially outside the arc at the same angle as the arc's midpoint
    theta_mid = np.pi / 4
    outside = 55.0 * np.array([np.cos(theta_mid)]), 55.0 * np.array([np.sin(theta_mid)])
    inside = 45.0 * np.array([np.cos(theta_mid)]), 45.0 * np.array([np.sin(theta_mid)])

    off_outside = project_to_frenet(outside[0], outside[1], cl)
    off_inside = project_to_frenet(inside[0], inside[1], cl)
    print(f"Arc test -> outside offset: {off_outside}, inside offset: {off_inside}")

    assert approx(abs(off_outside[0]), 5.0, tol=0.2), "Should measure ~5m radially outside"
    assert approx(abs(off_inside[0]), 5.0, tol=0.2), "Should measure ~5m radially inside"
    assert np.sign(off_outside[0]) != np.sign(off_inside[0]), "Inside/outside must have opposite signs"
    print("PASS: arc_offsets\n")

def test_centerline_picks_fastest_lap():
    from datetime import timedelta
    slow_traj = {'S': np.arange(0, 10, 1.0), 'X': np.arange(0, 10, 1.0) + 5, 'Y': np.zeros(10)}
    fast_traj = {'S': np.arange(0, 10, 1.0), 'X': np.arange(0, 10, 1.0), 'Y': np.zeros(10)}
    laps = [(timedelta(seconds=90), slow_traj), (timedelta(seconds=80), fast_traj)]
    cl = build_dense_centerline(laps)
    # Fastest lap's X starts at 0, slow lap's X starts at 5 -> centerline should track fast_traj
    assert approx(cl['X'][0], 0.0), "Centerline should be built from the fastest lap, not lap[0]"
    print("PASS: centerline_picks_fastest_lap\n")

if __name__ == "__main__":
    test_straight_line_offsets()
    test_arc_offsets()
    test_centerline_picks_fastest_lap()
    print("ALL SYNTHETIC TESTS PASSED")