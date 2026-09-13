import json
import numpy as np
import fastf1
from boundary_calibrator import calibrate_unit_scale, extract_frenet_trajectory, project_to_frenet


def project_to_frenet_local_window(x, y, s_query, centerline, window_m=15.0):
    """
    Same as project_to_frenet, but restricts the nearest-neighbor search to
    centerline points within `window_m` of the query point's own arc-length
    position. Prevents mismatches on chicanes/hairpins where the path folds
    back near itself in raw X/Y space.
    """
    cx, cy, cs = centerline['X'], centerline['Y'], centerline['S']
    cdx, cdy = centerline['dX'], centerline['dY']

    x, y, s_query = np.asarray(x), np.asarray(y), np.asarray(s_query)
    offsets = np.zeros(len(x))

    for i in range(len(x)):
        mask = np.abs(cs - s_query[i]) <= window_m
        if not mask.any():
            mask = np.ones(len(cs), dtype=bool)  # fallback: full search
        cx_w, cy_w, cdx_w, cdy_w = cx[mask], cy[mask], cdx[mask], cdy[mask]

        dists = np.sqrt((cx_w - x[i]) ** 2 + (cy_w - y[i]) ** 2)
        idx = np.argmin(dists)
        cross = cdx_w[idx] * (y[i] - cy_w[idx]) - cdy_w[idx] * (x[i] - cx_w[idx])
        sign = 1.0 if cross >= 0 else -1.0
        offsets[i] = dists[idx] * sign

    return offsets


def diagnose(boundary_file, year, session_name, session_type, start_dist, end_dist):
    with open(boundary_file) as f:
        config = json.load(f)
    scale = config['unit_scale']
    centerline = {k: np.array(v) for k, v in config['centerline'].items()}

    session = fastf1.get_session(year, session_name, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=True)

    deleted_laps = session.laps[session.laps['Deleted'] == True].dropna(subset=['DeletedReason'])

    print(f"{'Driver':<8}{'Lap':<6}{'IsAccurate':<12}{'TrackStat':<10}"
          f"{'Global offset':<16}{'LocalWin offset':<16}")

    for _, lap in deleted_laps.iterlaps():
        if "TRACK LIMITS" not in str(lap['DeletedReason']).upper():
            continue
        tel = lap.get_telemetry()
        traj = extract_frenet_trajectory(tel, start_dist, end_dist, scale)
        if traj is None:
            continue

        global_offsets = project_to_frenet(traj['X'], traj['Y'], centerline)
        local_offsets = project_to_frenet_local_window(traj['X'], traj['Y'], traj['S'], centerline)

        g_max = global_offsets[np.argmax(np.abs(global_offsets))]
        l_max = local_offsets[np.argmax(np.abs(local_offsets))]

        print(f"{lap['Driver']:<8}{int(lap['LapNumber']):<6}{str(lap['IsAccurate']):<12}"
              f"{str(lap['TrackStatus']):<10}{g_max:<16.3f}{l_max:<16.3f}")


if __name__ == "__main__":
    diagnose(
        boundary_file="boundaries/austria_2023_turn_9-10.json",
        year=2023, session_name='Austria', session_type='Q',
        start_dist=4100, end_dist=4250,
    )