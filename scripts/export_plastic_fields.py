"""Accumulate finite plastic volume and matrix storage fields along the path.

Plastic contributions start from Jp=1 at initialization and include every
accepted source/checkpoint ancestor. Each finite volume increment is split
in proportion to its simultaneous tensile and shear logarithmic volume
increments. The sum is Jp-1; the last-step multipliers are stored separately.
Matrix-water contributions use the same symmetric increments and 120 s
origin as export_mechanism_history.py, here retained element by element.

Spatial fields use reference coordinates. Regional integrals are delimited
by the crown (z=0) and invert (z=-D), which follow edges of this mesh.
"""

import numpy as np
import torch

from export_crack_transport import write_csv
from face_capacity_pinn import saturation
from fracture_physics import FractureProblem
from prepare_paper_results import OUTPUT, ROOT, read_path


SNAPSHOT_TIMES = (540., 540.038, 540.070, 540.071)


def plastic_volume_increment(old_j, tensile_log, shear_log):
    """Allocate a finite plastic volume increment to its two active modes."""
    total_log = tensile_log + shear_log
    factor = old_j.copy()
    np.divide(old_j * np.expm1(total_log), total_log, out=factor, where=total_log != 0)
    return factor * tensile_log, factor * shear_log


def main():
    segments = read_path()
    problem = FractureProblem(**segments[-1][2]['problem'])
    rows = [row for *_, part in segments for row in part]
    npsi = (1 + np.sin(np.deg2rad(problem.dilation_deg))) / (1 - np.sin(np.deg2rad(problem.dilation_deg)))
    snapshots, summaries = [], []
    old_jp = None
    baseline_water = None
    for row in rows:
        with np.load(ROOT / row['source_directory'] / row['checkpoint']) as field:
            if old_jp is None:
                xy, triangles, crack_cells = (field[key].copy() for key in ('xy', 'triangles', 'crack_cells'))
                points = xy[triangles]
                area = np.linalg.det(points[:, 1:] - points[:, :1]) / 2
                centers = points.mean(1)
                old_jp = np.ones(len(area))
                tensile = np.zeros(len(area))
                shear = tensile.copy()
                geometry_water, saturation_water, compression_water = (tensile.copy() for _ in range(3))
                regions = {'whole_domain': np.ones(len(area), dtype=bool),
                           'cover': centers[:, 1] >= 0,
                           'face_level': (centers[:, 1] < 0) & (centers[:, 1] >= -problem.shield_diameter_m),
                           'below_invert': centers[:, 1] < -problem.shield_diameter_m}
                bank_triangles = []
                for nodes in crack_cells:
                    banks = []
                    for a, b in nodes.reshape(2, 2):
                        (element,) = np.flatnonzero(np.any(triangles == a, axis=1) & np.any(triangles == b, axis=1))
                        banks.append(element)
                    bank_triangles.append(banks)
            tensile_log = field['bulk_tensile_multiplier']
            shear_log = field['bulk_shear_multiplier'] * (npsi - 1)
            dt, ds = plastic_volume_increment(old_jp, tensile_log, shear_log)
            tensile += dt
            shear += ds
            old_jp = field['bulk_plastic_J'].copy()
            j = field['bulk_J']
            p = field['pressure_pa'][triangles]
            s = saturation(torch.from_numpy(p), problem).numpy()
            if abs(row['time_s'] - 120.) < 1.e-8:
                baseline_water = field['bulk_water'].mean(1).copy()
            elif baseline_water is not None:
                geometry_water += ((previous_s + s) / 2).mean(1) * (j - previous_j)
                saturation_water += ((previous_j + j) / 2 - 1 + problem.porosity0) * (s - previous_s).mean(1)
                compression_water += problem.porosity0 * problem.water_compressibility_pa_inv * (p - previous_p).mean(1)
            if row is rows[-1] or any(abs(row['time_s'] - time) < 1.e-8 for time in SNAPSHOT_TIMES):
                snapshot = {
                    'time_s': row['time_s'], 'step_time_s': row['step_time_s'],
                    'source_field': f"{row['source_directory']}/{row['checkpoint']}",
                    'J': j.copy(), 'Jp': old_jp.copy(),
                    'tensile_volume_ratio': tensile.copy(), 'shear_volume_ratio': shear.copy(),
                    'last_tensile_log_increment': tensile_log.copy(),
                    'last_shear_log_volume_increment': shear_log.copy(),
                    'u_m': field['u_m'].copy(),
                    'matrix_geometry_water_ratio': geometry_water.copy(),
                    'matrix_saturation_water_ratio': saturation_water.copy(),
                    'matrix_compression_water_ratio': compression_water.copy(),
                    'matrix_water_change_ratio': field['bulk_water'].mean(1) - baseline_water,
                    'matrix_saturation': s.mean(1),
                }
                snapshots.append(snapshot)
                for region, mask in regions.items():
                    weights = area[mask] * 1.e6
                    summary = {'time_s': row['time_s'], 'region': region,
                               'source_field': snapshot['source_field'],
                               'total_volume_ml_per_m': weights @ (j[mask] - 1),
                               'plastic_volume_ml_per_m': weights @ (old_jp[mask] - 1),
                               'tensile_volume_ml_per_m': weights @ tensile[mask],
                               'shear_volume_ml_per_m': weights @ shear[mask]}
                    summary.update({key.replace('_ratio', '_ml_per_m'): weights @ snapshot[key][mask]
                                    for key in snapshot if key.startswith('matrix_') and key.endswith('_ratio')})
                    summaries.append(summary)
            previous_j, previous_s, previous_p = j.copy(), s, p
    arrays = {key: np.array([item[key] for item in snapshots]) for key in snapshots[0]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'plastic_spatial_fields.npz',
                        xy=xy, triangles=triangles, crack_cells=crack_cells,
                        area_m2=area, centers=centers, crack_bank_triangles=np.array(bank_triangles), **arrays)
    write_csv(OUTPUT / 'plastic_spatial_summary.csv', summaries)
    print(f'Accumulated {len(rows)} accepted states; wrote {len(snapshots)} spatial snapshots')
    for item in snapshots:
        difference = area @ (item['tensile_volume_ratio'] + item['shear_volume_ratio'] - item['Jp'] + 1) * 1.e6
        water_difference = area @ (item['matrix_geometry_water_ratio'] + item['matrix_saturation_water_ratio']
                                  + item['matrix_compression_water_ratio'] - item['matrix_water_change_ratio']) * 1.e6
        print(f"t={item['time_s']:.6f}: plastic partition difference={difference:.3g}, "
              f'water partition difference={water_difference:.3g} mL/m')


if __name__ == '__main__':
    main()
