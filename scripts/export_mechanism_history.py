"""Export the water partition along the accepted source/checkpoint history.

All volumes are mL per metre of out-of-plane thickness. ``cumulative_*``
columns start at the accepted 120 s state. ``stage_*`` columns start at the
accepted state immediately before each change in mean face pressure; that
state's time is ``stage_start_time_s``. The first row defines the baseline
and contains zero increments, without using its preceding time interval.

Matrix geometry, saturation and compression contributions are summed over
consecutive accepted states using the symmetric product increment of
W = (J - 1 + n0) S + n0 cw (p - p_initial). The matrix-water total is read
independently from saved W. Matrix quadrature uses reference triangle area
and equal nodal weights; fracture inventory uses reference segment length
and Gauss weights because saved Wf already contains the midline stretch.

Supply and face sweep use archived cumulative quantities; outflow integrates
each accepted end-step rate over its step time. ``bulk_geometric_volume``,
``bulk_plastic_volume`` and ``crack_geometric_volume`` are separate geometric
diagnostics, referenced to the undeformed skeleton or current cavity. They
are not additional water-budget terms. Every row retains its source field.
"""

import csv

import numpy as np
import torch

from face_capacity_pinn import saturation
from fracture_physics import FractureProblem
from prepare_paper_results import OUTPUT, ROOT, read_path


WATER_TERMS = (
    'supplied_volume', 'face_swept_volume', 'matrix_geometry',
    'matrix_saturation', 'matrix_compression', 'matrix_water_change',
    'fracture_water_change', 'outflow',
)


def read_state(row, problem, root=ROOT):
    """Read the fields and quadratures needed by the inventory calculation."""
    with np.load(root / row['source_directory'] / row['checkpoint']) as field:
        cells = field['triangles']
        points = field['xy'][cells]
        area = np.linalg.det(points[:, 1:] - points[:, :1]) / 2
        pressure = field['pressure_pa'][cells]
        weights = np.polynomial.legendre.leggauss(field['crack_water'].shape[1])[1] / 2
        return {
            'area': area,
            'J': field['bulk_J'][:, None],
            'S': saturation(torch.from_numpy(pressure), problem).numpy(),
            'p': pressure,
            'water': field['bulk_water'],
            'fracture_water': float(np.sum(
                field['crack_length_m'][:, None] * weights * field['crack_water'])) * 1.e6,
            'bulk_geometric_volume': float(area @ (field['bulk_J'] - 1)) * 1.e6,
            'bulk_plastic_volume': float(area @ (field['bulk_plastic_J'] - 1)) * 1.e6,
            'crack_geometric_volume': float(np.sum(
                field['crack_current_length'][:, None] * weights * field['crack_aperture'])) * 1.e6,
        }


def matrix_partition(previous, current, problem):
    """Signed reference-volume contributions for one accepted time interval."""
    j0, j1 = previous['J'], current['J']
    s0, s1 = previous['S'], current['S']
    contributions = (
        (s0 + s1) / 2 * (j1 - j0),
        (j0 + j1 - 2 + 2 * problem.porosity0) / 2 * (s1 - s0),
        problem.porosity0 * problem.water_compressibility_pa_inv * (current['p'] - previous['p']),
    )
    return np.array([np.sum(current['area'][:, None] / 3 * term) * 1.e6
                     for term in contributions])


def build_history(rows, problem, root=ROOT):
    """Accumulate partition increments without resetting them at restarts."""
    if not rows:
        raise ValueError('The reporting window contains no accepted state')
    first = rows[0]
    baseline = read_state(first, problem, root)
    previous, previous_row = baseline, first
    partition = np.zeros(3)
    outflow = 0.
    stage_start = first['time_s']
    stage_baseline = dict.fromkeys(WATER_TERMS, 0.)
    history = []
    for index, row in enumerate(rows):
        current = read_state(row, problem, root) if index else baseline
        elapsed = row['time_s'] - previous_row['time_s'] if index else 0.
        if index:
            if elapsed <= 0 or not np.isclose(elapsed, row['step_time_s'], rtol=1.e-6, atol=1.e-10):
                raise ValueError(f"Inconsistent accepted time interval at {row['source_directory']}/{row['checkpoint']}")
            if not np.isclose(row['face_mean_total_pressure_kpa'],
                              previous_row['face_mean_total_pressure_kpa'], rtol=0, atol=1.e-7):
                stage_start = previous_row['time_s']
                stage_baseline = {term: history[-1][f'cumulative_{term}_ml_per_m']
                                  for term in WATER_TERMS}
            partition += matrix_partition(previous, current, problem)
            outflow += row['outflow_ml_per_m_s'] * row['step_time_s']
        totals = {
            'supplied_volume': row['injected_volume_ml_per_m'] - first['injected_volume_ml_per_m'],
            'face_swept_volume': row['face_swept_volume_ml_per_m'] - first['face_swept_volume_ml_per_m'],
            'matrix_geometry': partition[0],
            'matrix_saturation': partition[1],
            'matrix_compression': partition[2],
            'matrix_water_change': float(np.sum(
                current['area'][:, None] / 3 * (current['water'] - baseline['water']))) * 1.e6,
            'fracture_water_change': current['fracture_water'] - baseline['fracture_water'],
            'outflow': outflow,
        }
        exported = {
            'time_s': row['time_s'],
            'face_mean_total_pressure_kpa': row['face_mean_total_pressure_kpa'],
            'mean_face_displacement_mm': row['mean_face_displacement_mm'],
            'interval_time_s': elapsed,
            'baseline_time_s': first['time_s'],
            'stage_start_time_s': stage_start,
            'stage_elapsed_time_s': row['time_s'] - stage_start,
        }
        exported.update({f'cumulative_{term}_ml_per_m': value for term, value in totals.items()})
        exported.update({f'stage_{term}_ml_per_m': value - stage_baseline[term]
                         for term, value in totals.items()})
        exported.update({f'{term}_ml_per_m': current[term] for term in (
            'fracture_water', 'bulk_geometric_volume', 'bulk_plastic_volume', 'crack_geometric_volume')})
        exported.update({key: row[key] for key in ('source_directory', 'checkpoint')})
        history.append(exported)
        previous, previous_row = current, row
    return history


def main():
    segments = read_path()
    rows = [row for _, _, _, part in segments for row in part if row['time_s'] >= 120 - 1.e-8]
    problem = FractureProblem(**segments[-1][2]['problem'])
    history = build_history(rows, problem)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / 'mechanism_history.csv'
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    held = next(row for row in history if abs(row['time_s'] - 540.) < 1.e-8)
    table = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'Time interval (s) & Geometry & Saturation & Compression & Net storage \\', r'\midrule']
    for start, end in ((history[0], held), (held, history[-1])):
        time_label = '--'.join(f"{row['time_s']:.6f}".rstrip('0').rstrip('.') for row in (start, end))
        values = [end[f'cumulative_matrix_{term}_ml_per_m'] - start[f'cumulative_matrix_{term}_ml_per_m']
                  for term in ('geometry', 'saturation', 'compression', 'water_change')]
        table.append(f'{time_label} & {values[0]:.4f} & {values[1]:.4f} & '
                     f'{values[2]:.6f} & {values[3]:.4f} ' + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (OUTPUT / 'matrix_storage_partition.tex').write_text('\n'.join(table) + '\n', encoding='utf-8')
    matrix_error = max(abs(sum(row[f'cumulative_matrix_{term}_ml_per_m']
                              for term in ('geometry', 'saturation', 'compression'))
                           - row['cumulative_matrix_water_change_ml_per_m']) for row in history)
    budget_error = max(abs(row['cumulative_supplied_volume_ml_per_m']
                           - sum(row[f'cumulative_{term}_ml_per_m'] for term in (
                               'face_swept_volume', 'matrix_water_change', 'fracture_water_change', 'outflow')))
                       for row in history)
    print(f'Wrote {len(history)} accepted states to {destination.relative_to(ROOT)}')
    print(f'Maximum matrix partition difference: {matrix_error:.3g} mL/m')
    print(f'Maximum cumulative water-balance difference: {budget_error:.3g} mL/m')


if __name__ == '__main__':
    main()
