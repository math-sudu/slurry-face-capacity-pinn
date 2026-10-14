"""Export fracture rates and representative profiles on the accepted path.

Rates are mL/(m s), volumes mL/m, and lengths mm. Positive edge flow runs
from crown to ground; positive wall exchange runs from crack to matrix.
Each pressure quadrature point owns a subcell with reference length L*wq.
Its physical wall measure is current length times wq. Saved crack_water
already includes the line stretch, whereas crack_storage_change is an
integrated subcell volume increment. Cumulative rates start at 120 s.

Profile edge locations use cumulative quadrature weights, the boundaries
of these control volumes; pressures and aperture retain the Gauss-point
locations used by the solver. Full-path profiles include the closed middle
and isolated surface opening, even when a plot focuses on the crown.
"""

import csv

import numpy as np

from prepare_paper_results import OUTPUT, ROOT, read_path


PROFILE_TIMES = (540., 540.030, 540.038, 540.066, 540.068, 540.070,
                 540.07025, 540.07075, 540.070875, 540.071)


def channel_fields(field, dt):
    """Recover local control-volume quantities without a nonlinear solve."""
    length = field['crack_length_m']
    current_length = field['crack_current_length']
    fraction = field['crack_quadrature_fraction']
    weights = np.polynomial.legendre.leggauss(len(fraction))[1] / 2
    shape = np.stack([1 - fraction, fraction], axis=1)
    cells = field['crack_cells']
    pm = field['pressure_pa'][cells]
    matrix_pressure = ((pm[:, :2] + pm[:, 2:]) / 2) @ shape.T
    wall = (current_length[:, None] * weights * field['crack_exchange']).ravel() * 1.e6
    storage = field['crack_storage_change'] * 1.e6 / dt
    flow = field['crack_edge_flux'] * 1.e6
    water = (length[:, None] * weights * field['crack_water']).ravel() * 1.e6
    coordinates = {}
    for prefix, lengths in (('reference', length), ('current', current_length)):
        start = np.r_[0., np.cumsum(lengths)[:-1]]
        coordinates[prefix + '_s_mm'] = (start[:, None] + lengths[:, None] * fraction).ravel() * 1000
        coordinates[prefix + '_edge_s_mm'] = np.r_[0., np.cumsum(
            (lengths[:, None] * weights).ravel())] * 1000
    return {
        **coordinates,
        'matrix_pressure_kpa': matrix_pressure.ravel() / 1000,
        'aperture_um': field['crack_aperture'].ravel() * 1.e6,
        'fracture_pressure_kpa': field['crack_pressure'].ravel() / 1000,
        'fracture_saturation': field['crack_saturation'].ravel(),
        'damage': field['crack_damage'].ravel(),
        'water_ml_per_m': water,
        'wall_rate_ml_per_m_s': wall,
        'storage_rate_ml_per_m_s': storage,
        'edge_rate_ml_per_m_s': flow,
    }


def write_csv(destination, rows):
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    segments = read_path()
    rows = [row for *_, part in segments for row in part if row['time_s'] >= 120 - 1.e-8]
    history, profiles = [], []
    accumulated = np.zeros(3)
    max_cell_defect, max_inventory_difference = 0., 0.
    previous_water = None
    for index, row in enumerate(rows):
        with np.load(ROOT / row['source_directory'] / row['checkpoint']) as field:
            data = channel_fields(field, row['step_time_s'])
            order = len(field['crack_quadrature_fraction'])
        flow, wall, storage = (data[key] for key in (
            'edge_rate_ml_per_m_s', 'wall_rate_ml_per_m_s', 'storage_rate_ml_per_m_s'))
        rates = np.array([flow[0], wall.sum(), flow[-1]])
        water = data['water_ml_per_m'].sum()
        if index:
            accumulated += rates * row['step_time_s']
            max_inventory_difference = max(max_inventory_difference, abs(
                water - previous_water - storage.sum() * row['step_time_s']))
        else:
            baseline_water = water
        max_cell_defect = max(max_cell_defect, float(np.max(np.abs(
            (storage + wall + np.diff(flow)) * row['step_time_s']))))
        provenance = {key: row[key] for key in ('time_s', 'source_directory', 'checkpoint')}
        history.append({
            'time_s': row['time_s'], 'step_time_s': row['step_time_s'],
            'face_pressure_kpa': row['face_mean_total_pressure_kpa'],
            'inlet_rate_ml_per_m_s': flow[0], 'outlet_rate_ml_per_m_s': flow[-1],
            'wall_rate_ml_per_m_s': wall.sum(), 'storage_rate_ml_per_m_s': storage.sum(),
            'matrix_to_crack_rate_ml_per_m_s': -np.minimum(wall, 0).sum(),
            'crack_to_matrix_rate_ml_per_m_s': np.maximum(wall, 0).sum(),
            'water_ml_per_m': water,
            'cumulative_inlet_ml_per_m': accumulated[0],
            'cumulative_wall_ml_per_m': accumulated[1],
            'cumulative_outlet_ml_per_m': accumulated[2],
            'water_change_ml_per_m': water - baseline_water,
            **provenance,
        })
        if row is rows[-1] or any(abs(row['time_s'] - time) < 1.e-8 for time in PROFILE_TIMES):
            for cell in range(len(wall)):
                profile = {**provenance, 'segment': cell // order + 1, 'subcell': cell % order + 1}
                for key, values in data.items():
                    if '_edge_' not in key and key != 'edge_rate_ml_per_m_s':
                        profile[key] = values[cell]
                for prefix in ('reference', 'current'):
                    for offset, side in ((0, 'upstream'), (1, 'downstream')):
                        profile[f'{prefix}_{side}_s_mm'] = data[prefix + '_edge_s_mm'][cell + offset]
                profile['upstream_rate_ml_per_m_s'] = flow[cell]
                profile['downstream_rate_ml_per_m_s'] = flow[cell + 1]
                profiles.append(profile)
        previous_water = water
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / 'crack_flow_history.csv', history)
    write_csv(OUTPUT / 'crack_profiles.csv', profiles)
    print(f'Wrote {len(history)} rate states and {len(profiles)} profile subcells')
    print(f'Maximum subcell water-balance defect: {max_cell_defect:.3g} mL/m')
    print(f'Maximum saved-increment versus inventory difference: {max_inventory_difference:.3g} mL/m')
    print(f'Cumulative rate versus inventory difference: '
          f'{accumulated[0] - accumulated[1] - accumulated[2] - water + baseline_water:.3g} mL/m')


if __name__ == '__main__':
    main()
