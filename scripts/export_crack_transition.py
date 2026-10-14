"""Resolve filling, traction and opening rates through crown debonding.

Each row is one interface segment at one accepted state, starting at 540.066 s.
Segment averages use the solver's Gauss weights. Hydraulic opening force is
the integral of Sf*pf over current length; cohesive closing force is its
nominal counterpart over reference length, recovered from the saved net
traction. Their difference is the normal separation drive on that segment.
Opening and plastic-volume rates use the actual accepted increment, including
changes of time step. Bank plastic rates refer to the two adjoining triangles.
Filling is aperture-weighted coverage, undefined for a closed segment.
"""

import argparse
from pathlib import Path

import numpy as np

from export_crack_transport import channel_fields, write_csv
from prepare_paper_results import FINAL, OUTPUT, ROOT, read_path


def segment_response(field, previous, dt):
    """Integrate the saved local hydraulic and mechanical quantities."""
    aperture = field['crack_aperture']
    order = aperture.shape[1]
    weights = np.polynomial.legendre.leggauss(order)[1] / 2
    length = field['crack_length_m']
    current_length = field['crack_current_length']
    pressure = field['crack_pressure']
    coverage = field['crack_saturation']
    hydraulic = current_length * ((coverage * pressure) @ weights)
    cohesive = length * (field['crack_traction'][..., 0] @ weights) + hydraulic
    channel = channel_fields(field, dt)
    geometry = current_length * (aperture @ weights) * 1.e6
    water = channel['water_ml_per_m'].reshape(-1, order).sum(1)
    filling = np.full(len(length), np.nan)
    np.divide((aperture * coverage) @ weights, aperture @ weights,
              out=filling, where=(aperture @ weights) > 0)
    return {
        'mean_aperture_um': aperture @ weights * 1.e6,
        'mean_opening_rate_mm_s': (aperture - previous['crack_aperture']) @ weights / dt * 1000,
        'mean_pressure_kpa': pressure @ weights / 1000,
        'mean_effective_pressure_kpa': (coverage * pressure) @ weights / 1000,
        'downstream_pressure_kpa': pressure[:, -1] / 1000,
        'mean_coverage': coverage @ weights,
        'cavity_filling_fraction': filling,
        'downstream_coverage': coverage[:, -1],
        'minimum_damage': field['crack_damage'].min(1),
        'geometric_volume_ml_per_m': geometry,
        'water_ml_per_m': water,
        'inlet_rate_ml_per_m_s': channel['edge_rate_ml_per_m_s'][::order][:-1],
        'outlet_rate_ml_per_m_s': channel['edge_rate_ml_per_m_s'][::order][1:],
        'wall_rate_ml_per_m_s': channel['wall_rate_ml_per_m_s'].reshape(-1, order).sum(1),
        'storage_rate_ml_per_m_s': channel['storage_rate_ml_per_m_s'].reshape(-1, order).sum(1),
        'hydraulic_opening_force_n_per_m': hydraulic,
        'cohesive_closing_force_n_per_m': cohesive,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final', type=Path, default=FINAL)
    args = parser.parse_args()
    segments = read_path(args.final)
    rows = [row for *_, part in segments for row in part]
    records = []
    previous = None
    for row in rows:
        if row['time_s'] < 540.064 - 1.e-8:
            continue
        with np.load(ROOT / row['source_directory'] / row['checkpoint']) as archive:
            field = {key: archive[key] for key in archive.files}
        if previous is None:
            xy, triangles = field['xy'], field['triangles']
            points = xy[triangles]
            area = np.linalg.det(points[:, 1:] - points[:, :1]) / 2
            bank_triangles = []
            for nodes in field['crack_cells']:
                banks = []
                for a, b in nodes.reshape(2, 2):
                    (element,) = np.flatnonzero(np.any(triangles == a, axis=1)
                                              & np.any(triangles == b, axis=1))
                    banks.append(element)
                bank_triangles.append(banks)
            bank_triangles = np.array(bank_triangles)
        elif row['time_s'] >= 540.066 - 1.e-8:
            dt = row['step_time_s']
            local = segment_response(field, previous, dt)
            plastic_rate = (field['bulk_plastic_J'] - previous['bulk_plastic_J']) / dt
            for segment in range(len(field['crack_length_m'])):
                banks = bank_triangles[segment]
                record = {
                    'time_s': row['time_s'], 'step_time_s': dt, 'segment': segment + 1,
                    'face_displacement_mm': row['mean_face_displacement_mm'],
                    'max_velocity_m_s': row['max_velocity_m_s'],
                    'kinetic_energy_j_per_m': row['kinetic_energy_j_per_m'],
                    'supply_rate_ml_per_m_s': row['injection_increment_ml_per_m'] / dt,
                    'face_sweep_rate_ml_per_m_s': (float(field['face_swept_volume_m2'])
                        - float(previous['face_swept_volume_m2'])) * 1.e6 / dt,
                    'bulk_plastic_rate_ml_per_m_s': area @ plastic_rate * 1.e6,
                    'bank_plastic_rate_ml_per_m_s': area[banks] @ plastic_rate[banks] * 1.e6,
                    'bank_tensile_log_rate_s_inv': field['bulk_tensile_multiplier'][banks].max() / dt,
                    'bank_shear_multiplier_rate_s_inv': field['bulk_shear_multiplier'][banks].max() / dt,
                    **{key: value[segment] for key, value in local.items()},
                    'source_directory': row['source_directory'], 'checkpoint': row['checkpoint'],
                }
                records.append(record)
        previous = field
    write_csv(OUTPUT / 'crack_transition.csv', records)
    print(f'Wrote {len(records)} segment states to crack_transition.csv')


if __name__ == '__main__':
    main()
