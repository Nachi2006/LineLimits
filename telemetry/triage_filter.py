import os
import json
import numpy as np
import pandas as pd
import fastf1


class FastF1TriageFilter:
    def __init__(self, boundary_file):
        if not os.path.exists(boundary_file):
            raise FileNotFoundError(f"Boundary file {boundary_file} not found.")

        with open(boundary_file, 'r') as f:
            self.config = json.load(f)

        self.unit_scale = self.config['unit_scale']
        self.boundary_pos_m = self.config['boundary_pos_m']
        self.boundary_neg_m = self.config['boundary_neg_m']
        self.std_pos_m = self.config['std_pos_m']
        self.std_neg_m = self.config['std_neg_m']
        self.start_dist = self.config['start_dist']
        self.end_dist = self.config['end_dist']

        if not self.config.get('ground_truth_calibrated', True):
            print(f"WARNING: boundary for {self.config.get('corner')} was not calibrated "
                  f"against real deletions on at least one side. Treat as provisional.")

        self.centerline = {
            'S': np.array(self.config['centerline']['S']),
            'X': np.array(self.config['centerline']['X']),
            'Y': np.array(self.config['centerline']['Y']),
            'dX': np.array(self.config['centerline']['dX']),
            'dY': np.array(self.config['centerline']['dY']),
        }

    def _extract_frenet_trajectory(self, tel):
        mask = (tel['Distance'] >= self.start_dist) & (tel['Distance'] <= self.end_dist)
        seg = tel[mask]
        if len(seg) < 2:
            return None
        return {
            'X': seg['X'].values / self.unit_scale,
            'Y': seg['Y'].values / self.unit_scale,
            'Speed': seg['Speed'].values,
        }

    def _project_to_frenet(self, x, y):
        cx, cy = self.centerline['X'], self.centerline['Y']
        cdx, cdy = self.centerline['dX'], self.centerline['dY']

        x, y = np.asarray(x), np.asarray(y)
        dists = np.sqrt((cx[None, :] - x[:, None]) ** 2 + (cy[None, :] - y[:, None]) ** 2)
        idx = np.argmin(dists, axis=1)
        base_dist = dists[np.arange(len(x)), idx]

        cross = cdx[idx] * (y - cy[idx]) - cdy[idx] * (x - cx[idx])
        sign = np.where(cross >= 0, 1.0, -1.0)
        return base_dist * sign

    def triage_lap(self, lap):
        """
        Evaluates a lap and returns (Triage_Tier, metrics_dict).

        Safety principle: whenever we cannot confidently evaluate a lap
        (missing telemetry, non-green track status, etc.) we route it
        onward rather than assuming it's clean — a wrong "send to CV" costs
        compute, a wrong "drop" costs a missed violation.
        """
        status = str(lap['TrackStatus'])
        if any(c in status for c in ['4', '5', '6', '7']):
            return 'NOT_EVALUATED_TRACK_STATUS', {
                'reason': f'Track status not clear: {status}',
                'note': 'SC/VSC/red flag — track limits enforcement differs; not auto-cleared.',
            }

        tel = lap.get_telemetry()
        traj = self._extract_frenet_trajectory(tel)
        if traj is None:
            return 'NOT_EVALUATED_MISSING_TELEMETRY', {
                'reason': 'Insufficient telemetry in corner window',
                'note': 'Cannot confirm legality from telemetry alone; route for manual/CV check.',
            }

        offsets = self._project_to_frenet(traj['X'], traj['Y'])
        max_pos = offsets.max() if offsets.max() > 0 else 0.0
        max_neg = offsets.min() if offsets.min() < 0 else 0.0

        avg_speed_kph = float(np.mean(traj['Speed']))
        avg_speed_mps = avg_speed_kph / 3.6
        gps_uncertainty_m = avg_speed_mps * (1.0 / 3.7)  # distance between GPS fixes

        buffer_pos_m = max(self.std_pos_m, gps_uncertainty_m)
        buffer_neg_m = max(self.std_neg_m, gps_uncertainty_m)

        metrics = {
            'max_pos_offset': float(max_pos),
            'max_neg_offset': float(max_neg),
            'boundary_pos': self.boundary_pos_m,
            'boundary_neg': self.boundary_neg_m,
            'buffer_pos': buffer_pos_m,
            'buffer_neg': buffer_neg_m,
            'avg_speed': avg_speed_kph,
            'gps_uncertainty': gps_uncertainty_m,
        }

        # Evaluate each side independently against its own calibrated boundary
        pos_tier = self._classify(max_pos, self.boundary_pos_m, buffer_pos_m, positive=True)
        neg_tier = self._classify(max_neg, self.boundary_neg_m, buffer_neg_m, positive=False)

        # Worst-case side wins (violation-priority ordering)
        priority = {'DEEP_VIOLATION_CV_LOW_PRIORITY': 3, 'AMBIGUOUS_CV_HIGH_PRIORITY': 2, 'DEEP_CLEAN': 1}
        tier = max([pos_tier, neg_tier], key=lambda t: priority[t])
        return tier, metrics

    @staticmethod
    def _classify(offset, boundary, buffer_m, positive):
        if positive:
            if offset < boundary - buffer_m:
                return 'DEEP_CLEAN'
            elif offset > boundary + buffer_m:
                return 'DEEP_VIOLATION_CV_LOW_PRIORITY'
            return 'AMBIGUOUS_CV_HIGH_PRIORITY'
        else:
            if offset > boundary + buffer_m:  # boundary is negative; "inside" it means less negative
                return 'DEEP_CLEAN'
            elif offset < boundary - buffer_m:
                return 'DEEP_VIOLATION_CV_LOW_PRIORITY'
            return 'AMBIGUOUS_CV_HIGH_PRIORITY'


def run_triage_simulation():
    print("--- Running Triage Filter Simulation ---")
    filt = FastF1TriageFilter("boundaries/austria_2023_turn_9-10.json")
    print(f"Loaded boundary. Positive bound: {filt.boundary_pos_m:.3f}m, "
          f"Negative bound: {filt.boundary_neg_m:.3f}m")

    session = fastf1.get_session(2023, 'Austria', 'Q')
    session.load(telemetry=True, laps=True, weather=False, messages=False)

    triage_counts = {}
    laps_to_check = session.laps.pick_drivers(['VER', 'PER', 'HAM', 'NOR'])

    for _, lap in laps_to_check.iterlaps():
        if pd.isna(lap['LapTime']):
            continue

        driver = lap['Driver']
        tier, metrics = filt.triage_lap(lap)
        triage_counts[tier] = triage_counts.get(tier, 0) + 1

        if tier != 'DEEP_CLEAN':
            print(f"[{tier}] {driver} Lap {lap['LapNumber']}: {metrics}")

    print("\nSimulation Summary:")
    for tier, count in triage_counts.items():
        print(f"  {tier}: {count} laps")


if __name__ == "__main__":
    run_triage_simulation()