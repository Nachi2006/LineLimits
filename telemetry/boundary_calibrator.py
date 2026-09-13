import os
import json
import numpy as np
import pandas as pd
import fastf1


def calibrate_unit_scale(session):
    """
    Derives the multiplier to convert FastF1 X/Y units to meters, using the
    ratio of raw X/Y step size to the corresponding step in the `Distance`
    channel (which is already in meters) over several representative laps.
    """
    ratios = []
    valid_laps = session.laps.pick_quicklaps().dropna(subset=['LapTime'])
    for _, lap in valid_laps.head(5).iterlaps():
        tel = lap.get_telemetry()
        if tel.empty or 'X' not in tel or 'Y' not in tel or 'Distance' not in tel:
            continue
        dx = np.diff(tel['X'].values)
        dy = np.diff(tel['Y'].values)
        ds = np.diff(tel['Distance'].values)
        mask = ds > 0.5  # avoid div-by-zero / lap-wrap artifacts
        hypot = np.sqrt(dx[mask] ** 2 + dy[mask] ** 2)
        ratios.extend((hypot / ds[mask]).tolist())

    if not ratios:
        print("WARNING: Could not calibrate unit scale. Defaulting to 10.0")
        return 10.0
    return float(np.median(ratios))


def extract_frenet_trajectory(tel, start_dist, end_dist, scale):
    mask = (tel['Distance'] >= start_dist) & (tel['Distance'] <= end_dist)
    seg = tel[mask]
    if len(seg) < 2:
        return None
    return {
        'X': seg['X'].values / scale,
        'Y': seg['Y'].values / scale,
        'S': seg['Distance'].values,
        'Speed': seg['Speed'].values,
    }


def build_dense_centerline(clean_laps_with_traj):
    """
    Uses the FASTEST clean lap's trajectory (by LapTime) as the reference
    centerline, upsampled to dense 0.1m spacing. Using the fastest lap avoids
    picking an arbitrary/off-pace line as the reference.
    """
    if not clean_laps_with_traj:
        return None

    fastest = min(clean_laps_with_traj, key=lambda item: item[0])
    t = fastest[1]

    sort_idx = np.argsort(t['S'])
    s_sorted, x_sorted, y_sorted = t['S'][sort_idx], t['X'][sort_idx], t['Y'][sort_idx]

    dense_s = np.arange(s_sorted.min(), s_sorted.max(), 0.1)
    dense_x = np.interp(dense_s, s_sorted, x_sorted)
    dense_y = np.interp(dense_s, s_sorted, y_sorted)

    return {
        'S': dense_s,
        'X': dense_x,
        'Y': dense_y,
        'dX': np.gradient(dense_x),
        'dY': np.gradient(dense_y),
    }


def project_to_frenet(x, y, centerline):
    """
    Vectorized nearest-point projection onto the centerline.
    Returns SIGNED lateral offsets: positive = one side of centerline
    (by convention, "outside" if the tangent is oriented with the direction
    of travel), negative = the other side. Sign must be preserved by callers
    since track limits are NOT symmetric left/right.
    """
    cx, cy = centerline['X'], centerline['Y']
    cdx, cdy = centerline['dX'], centerline['dY']

    x, y = np.asarray(x), np.asarray(y)
    dists = np.sqrt((cx[None, :] - x[:, None]) ** 2 + (cy[None, :] - y[:, None]) ** 2)
    idx = np.argmin(dists, axis=1)
    base_dist = dists[np.arange(len(x)), idx]

    cross = cdx[idx] * (y - cy[idx]) - cdy[idx] * (x - cx[idx])
    sign = np.where(cross >= 0, 1.0, -1.0)
    return base_dist * sign


def calibrate_boundary(year, session_name, session_type, corner_name, start_dist, end_dist):
    print(f"--- Calibrating Boundary for {session_name} {year} ({corner_name}) ---")

    session = fastf1.get_session(year, session_name, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=True)

    scale = calibrate_unit_scale(session)
    print(f"Calibrated Unit Scale: {scale:.4f} FastF1 units/meter")

    all_laps = session.laps
    clean_mask = (
        (all_laps['Deleted'] == False)
        & (all_laps['IsAccurate'] == True)
        & (all_laps['TrackStatus'] == '1')
    )
    clean_laps = all_laps[clean_mask]

    clean_laps_with_traj = []  # (LapTime, trajectory) for centerline selection
    for _, lap in clean_laps.iterlaps():
        tel = lap.get_telemetry()
        traj = extract_frenet_trajectory(tel, start_dist, end_dist, scale)
        if traj is not None:
            clean_laps_with_traj.append((lap['LapTime'], traj))

    print(f"Extracted {len(clean_laps_with_traj)} clean trajectories.")

    centerline = build_dense_centerline(clean_laps_with_traj)
    if centerline is None:
        raise RuntimeError("No valid clean trajectories found; cannot calibrate boundary.")

    # Signed max deviation per lap, kept separate by side (asymmetric track limits)
    pos_devs, neg_devs = [], []
    for _, t in clean_laps_with_traj:
        offsets = project_to_frenet(t['X'], t['Y'], centerline)
        pos_devs.append(offsets.max() if offsets.max() > 0 else 0.0)
        neg_devs.append(offsets.min() if offsets.min() < 0 else 0.0)

    pos_devs, neg_devs = np.array(pos_devs), np.array(neg_devs)

    def trim_and_summarize(devs, keep_lower_tail=True):
        """Trim the widest 2% (likely uncalled violations), return (bound, std_m)."""
        sorted_devs = np.sort(devs)
        cutoff = int(len(sorted_devs) * 0.98)
        trimmed = sorted_devs[:cutoff] if keep_lower_tail else sorted_devs[-cutoff:]
        bound = trimmed.max() if keep_lower_tail else trimmed.min()
        return float(bound), float(np.std(trimmed))

    pos_boundary, pos_std_m = trim_and_summarize(pos_devs, keep_lower_tail=True)
    neg_boundary, neg_std_m = trim_and_summarize(-neg_devs, keep_lower_tail=True)
    neg_boundary = -neg_boundary  # restore sign

    print(f"Historical boundary (positive side, 98th pct): {pos_boundary:.3f}m (std {pos_std_m:.3f}m)")
    print(f"Historical boundary (negative side, 98th pct): {neg_boundary:.3f}m (std {neg_std_m:.3f}m)")

    # Ground-truth calibration against actual track-limits deletions, per side
    deleted_laps = all_laps[all_laps['Deleted'] == True].dropna(subset=['DeletedReason'])
    pos_deletions, neg_deletions = [], []
    for _, lap in deleted_laps.iterlaps():
        if "TRACK LIMITS" not in str(lap['DeletedReason']).upper():
            continue
        tel = lap.get_telemetry()
        traj = extract_frenet_trajectory(tel, start_dist, end_dist, scale)
        if traj is None:
            continue
        offsets = project_to_frenet(traj['X'], traj['Y'], centerline)
        if offsets.max() > 0:
            pos_deletions.append(offsets.max())
        if offsets.min() < 0:
            neg_deletions.append(offsets.min())

    calibrated = True
    if pos_deletions:
        pos_min_deletion = min(pos_deletions)
        print(f"Positive side: {len(pos_deletions)} deletions, min distance {pos_min_deletion:.3f}m")
    else:
        pos_min_deletion = pos_boundary + 0.5
        calibrated = False
        print("WARNING: no positive-side track-limits deletions found. "
              "Using UNVALIDATED fallback boundary — do not treat this corner side as live-ready "
              "until calibrated against real deletions.")

    if neg_deletions:
        neg_min_deletion = max(neg_deletions)  # closest-to-zero = smallest magnitude violation
        print(f"Negative side: {len(neg_deletions)} deletions, min |distance| {abs(neg_min_deletion):.3f}m")
    else:
        neg_min_deletion = neg_boundary - 0.5
        calibrated = False
        print("WARNING: no negative-side track-limits deletions found. "
              "Using UNVALIDATED fallback boundary — do not treat this corner side as live-ready "
              "until calibrated against real deletions.")

    output = {
        'circuit': session_name,
        'year': year,
        'corner': corner_name,
        'start_dist': start_dist,
        'end_dist': end_dist,
        'unit_scale': scale,
        'boundary_pos_m': pos_min_deletion,
        'boundary_neg_m': neg_min_deletion,
        'std_pos_m': pos_std_m,
        'std_neg_m': neg_std_m,
        'ground_truth_calibrated': calibrated,
        'centerline': {k: v.tolist() for k, v in centerline.items()},
    }

    os.makedirs('boundaries', exist_ok=True)
    filename = f"boundaries/{session_name}_{year}_{corner_name}.json".replace(" ", "_").lower()
    with open(filename, 'w') as f:
        json.dump(output, f)
    print(f"Boundary configuration saved to {filename}")


if __name__ == "__main__":
    calibrate_boundary(
        year=2023,
        session_name='Austria',
        session_type='Q',
        corner_name='Turn 9-10',
        start_dist=4100,
        end_dist=4250,
    )