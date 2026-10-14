"""Recover radius evolution and the field effect of accepted feature updates.

The two local comparisons reconstruct the output-equilibrated, fixed-radius
state within an existing increment. Both sides use the same preceding physical
history, time step and displacement predictor. The adaptive side is the saved
accepted state, not a new loading path.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from prepare_paper_results import OUTPUT, ROOT, read_path
from pressure_dynamics import restore_dynamic_model
from fracture_physics import cohesive_normal

EVIDENCE = OUTPUT


def write_rows(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def history_data(rows, net):
    output = []
    previous = None
    for row in rows:
        with np.load(ROOT / row['source_directory'] / row['checkpoint']) as saved:
            logs = saved['parameters'][net.outputs:net.outputs + net.hidden]
        ratio = np.exp(logs)
        before, after = row['force_loss_before_feature_learning'], row['neural_loss']
        output.append({
            'time_s': row['time_s'],
            'pressure_kpa': row['face_mean_total_pressure_kpa'],
            'feature_updates': row['feature_updates'],
            'loss_before': before, 'loss_after': after,
            'loss_reduction_percent': 100 * (before - after) / before,
            'radius_ratio_p05': np.quantile(ratio, .05),
            'radius_ratio_median': np.median(ratio),
            'radius_ratio_p95': np.quantile(ratio, .95),
            'radius_ratio_min': ratio.min(), 'radius_ratio_max': ratio.max(),
            'max_log_radius_increment': np.nan if previous is None else np.max(np.abs(logs - previous)),
            'source_directory': row['source_directory'], 'checkpoint': row['checkpoint'],
        })
        previous = logs.copy()
    return output


def force_field(model, force):
    full = np.zeros(3 * model.system.n)
    full[model.system.free[model.mechanical]] = force
    return full.reshape(-1, 3)[:, :2]


def compare_increment(previous, row):
    model, _ = restore_dynamic_model(ROOT / previous['source_directory'],
                                     previous['checkpoint'], row['step_time_s'])
    system, net = model.system, model.network
    system.target = 1000 * row['face_mean_total_pressure_kpa']
    settings = json.loads((ROOT / row['source_directory'] / 'result.json').read_text(encoding='utf-8'))['settings']
    if settings.get('predict_velocity', False):
        model.predict_displacement()
    training = model.train(0)
    fixed_row, fixed = system.inspect()
    state, residual, _, _ = model.project(net.evaluate(net.parameters), tangent=False)
    fixed_force = model.forces(state, residual)
    fixed_parameters = net.parameters.copy()
    with np.load(ROOT / row['source_directory'] / row['checkpoint']) as saved:
        net.parameters = saved['parameters'].copy()
    system.state, residual, _, _ = model.project(net.evaluate(net.parameters), tangent=False)
    adapted_force = model.forces(system.state, residual)
    adapted_row, adapted = system.inspect()
    for observations, fields in ((fixed_row, fixed), (adapted_row, adapted)):
        effective = torch.as_tensor(fields['crack_effective_endpoints'][0, 0])
        maximum = torch.as_tensor(fields['crack_history_maximum'][0])
        observations['mouth_aperture_mm'] = 1000 * float(cohesive_normal(
            effective, torch.zeros_like(effective), maximum, system.b)[1])
        observations['bulk_plastic_volume_ml_per_m'] = 1.e6 * float(
            system.mesh.area.numpy() @ (fields['bulk_plastic_J'] - 1))
        observations['crack_water_ml_per_m'] = observations['fracture_water_ml_per_m']
    fixed_loss = float(np.sum((fixed_force / model.force_scale) ** 2))
    adapted_loss = float(np.sum((adapted_force / model.force_scale) ** 2))
    # These compare reconstructed identities with their saved counterparts,
    # not a physical-accuracy threshold or a training stopping requirement.
    np.testing.assert_allclose(fixed_loss, row['force_loss_before_feature_learning'], rtol=1.e-6)
    np.testing.assert_allclose(adapted_loss, row['neural_loss'], rtol=1.e-6)
    metrics = {
        'time_s': row['time_s'], 'pressure_kpa': row['face_mean_total_pressure_kpa'],
        'step_time_s': row['step_time_s'], 'predict_velocity': settings.get('predict_velocity', False),
        'loss_fixed': fixed_loss, 'loss_adapted': adapted_loss,
        'loss_reduction_percent': 100 * (fixed_loss - adapted_loss) / fixed_loss,
        'max_displacement_correction_um': 1.e6 * np.linalg.norm(adapted['u_m'] - fixed['u_m'], axis=1).max(),
        'max_matrix_pressure_change_kpa': np.max(np.abs(adapted['pressure_pa'] - fixed['pressure_pa'])) / 1000,
        'fixed_output_iterations': training['neural_iterations'],
        'previous_directory': previous['source_directory'], 'previous_checkpoint': previous['checkpoint'],
        'source_directory': row['source_directory'], 'checkpoint': row['checkpoint'],
    }
    for key in ('mean_face_displacement_mm', 'mouth_aperture_mm', 'crack_water_ml_per_m',
                'bulk_plastic_volume_ml_per_m'):
        metrics[key + '_fixed'] = fixed_row[key]
        metrics[key + '_adapted'] = adapted_row[key]
        np.testing.assert_allclose(adapted_row[key], row[key], rtol=1.e-6, atol=1.e-10)
    arrays = {'xy': fixed['xy'], 'triangles': fixed['triangles'],
              'force_fixed': force_field(model, fixed_force),
              'force_adapted': force_field(model, adapted_force),
              'scaled_force_fixed': force_field(model, fixed_force / model.force_scale),
              'scaled_force_adapted': force_field(model, adapted_force / model.force_scale),
              'u_fixed': fixed['u_m'], 'u_adapted': adapted['u_m'],
              'pressure_fixed': fixed['pressure_pa'], 'pressure_adapted': adapted['pressure_pa'],
              'parameters_fixed': fixed_parameters, 'parameters_adapted': net.parameters.copy()}
    return metrics, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-only', action='store_true')
    args = parser.parse_args()
    rows = [row for _, _, _, part in read_path() for row in part]
    last = rows[-1]
    model, _ = restore_dynamic_model(ROOT / last['source_directory'], last['checkpoint'], last['step_time_s'])
    write_rows(OUTPUT / 'neural_adaptation.csv', history_data(rows, model.network))
    if args.history_only:
        return
    torch.set_num_threads(2)
    metrics, arrays = [], {}
    with threadpool_limits(limits=2):
        for pressure in (22.5, 25.):
            index = max((i for i, row in enumerate(rows) if row['face_mean_total_pressure_kpa'] == pressure),
                        key=lambda i: 1 - rows[i]['neural_loss'] / rows[i]['force_loss_before_feature_learning'])
            print(f'Reconstructing fixed-radius state at {rows[index]["time_s"]:.9f} s', flush=True)
            comparison, fields = compare_increment(rows[index - 1], rows[index])
            metrics.append(comparison)
            arrays.update({f'p{pressure:g}_{key}': value for key, value in fields.items()})
            print(json.dumps(comparison), flush=True)
    EVIDENCE.mkdir(exist_ok=True)
    np.savez_compressed(EVIDENCE / 'feature_comparison.npz', **arrays)
    write_rows(OUTPUT / 'feature_comparison.csv', metrics)


if __name__ == '__main__':
    main()
