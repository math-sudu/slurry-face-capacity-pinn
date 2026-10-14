"""Read physical crack connectivity and Liu's surface gauges from accepted fields.

The displacement jump and initial traction are affine along each crack segment.
Their accepted history therefore determines the open intervals exactly, even
between Gauss points. This is an observation of the prescribed cohesive path;
bulk tensile plasticity is reported separately and never converted to aperture.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from fracture_physics import FractureProblem, cohesive_normal


def positive_interval(endpoints, threshold=0.):
    """Closure of the interval where an affine function exceeds a threshold."""
    left, right = np.asarray(endpoints)-threshold
    if max(left, right) <= 0:
        return None
    if min(left, right) >= 0:
        return (0., 1.)
    crossing = float(-left/(right-left))
    return (0., crossing) if left > 0 else (crossing, 1.)


def union_intervals(intervals):
    result = []
    for left, right in sorted(intervals):
        # Intervals that only touch share a zero-aperture point, which still
        # interrupts a hydraulic slit. Merge only an overlapping open set.
        if result and left < result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], right))
        else:
            result.append((left, right))
    return result


class CrackOpeningTracker:
    def __init__(self, b, xy, crack_cells):
        self.b = b
        self.cells = np.asarray(crack_cells)
        self.nodes = np.asarray(xy)[self.cells[:, :2]]
        tangent = self.nodes[:, 1]-self.nodes[:, 0]
        self.length = np.linalg.norm(tangent, axis=1)
        tangent /= self.length[:, None]
        self.normal = np.c_[tangent[:, 1], -tangent[:, 0]]
        vertical = -b.density_kg_m3*b.gravity_m_s2*(b.cover_m-self.nodes[:, :, 1])
        self.traction0 = (b.k0*self.normal[:, 0]**2+self.normal[:, 1]**2)[:, None]*vertical+b.saturation0*b.pressure0
        self.maximum = torch.zeros(self.traction0.shape)
        self.damaged = [[] for _ in self.cells]

    def update(self, displacement):
        b = self.b
        u = np.asarray(displacement)[self.cells]
        jumps = u[:, 2:]-u[:, :2]
        nodes, length, normal = self.nodes, self.length, self.normal
        if b.interface_geometry == 'corotated':
            nodes = self.nodes+(u[:, :2]+u[:, 2:])/2
            tangent = nodes[:, 1]-nodes[:, 0]
            length = np.linalg.norm(tangent, axis=1)
            tangent = tangent/length[:, None]
            normal = np.c_[tangent[:, 1], -tangent[:, 0]]
        jump = np.einsum('eji,ei->ej', jumps, normal)
        effective = jump+self.traction0/b.interface_stiffness_pa_m
        _, aperture, _, self.maximum = cohesive_normal(
            torch.tensor(jump), torch.tensor(self.traction0), self.maximum, b)
        aperture = aperture.numpy()
        opening = []
        for i, values in enumerate(effective):
            newly_damaged = positive_interval(values, b.tensile_strength_pa/b.interface_stiffness_pa_m)
            if newly_damaged is not None:
                self.damaged[i] = union_intervals(self.damaged[i]+[newly_damaged])
            positive = positive_interval(values)
            intervals = []
            if positive is not None:
                for left, right in self.damaged[i]:
                    lo, hi = max(left, positive[0]), min(right, positive[1])
                    if lo < hi:
                        intervals.append((lo, hi))
            opening.append(intervals)

        connected_length = 0.
        tip = nodes[0, 0].copy()
        reaches_surface = True
        for i, intervals in enumerate(opening):
            if aperture[i, 0] <= 0 or not intervals or intervals[0][0] != 0:
                reaches_surface = False
                break
            fraction = intervals[0][1]
            connected_length += fraction*length[i]
            tip = nodes[i, 0]+fraction*(nodes[i, 1]-nodes[i, 0])
            if fraction < 1 or aperture[i, 1] <= 0:
                reaches_surface = False
                break
        return {
            'connected_open_length_m': float(connected_length),
            'connected_open_tip_z_m': float(tip[1]),
            'open_path_to_surface': reaches_surface,
            'mouth_aperture_mm': float(aperture[0, 0])*1000,
            'surface_endpoint_aperture_mm': float(aperture[-1, 1])*1000,
            'surface_vertical_jump_mm': float(jumps[-1, 1, 1])*1000,
            'open_length_m': float(sum(length[i]*sum(hi-lo for lo, hi in intervals)
                                       for i, intervals in enumerate(opening))),
        }


def surface_uplift(xy, displacement, crack_cells, positions):
    """Piecewise linear surface trace, retaining both sides of its crack intersection.

    A gauge exactly on the discontinuity has two traces and needs an explicit
    footprint; choosing or averaging them silently would invent an observation.
    """
    xy, displacement = np.asarray(xy), np.asarray(displacement)
    minus, plus = np.asarray(crack_cells)[-1, [1, 3]]
    surface_x, surface_z = xy[minus]
    top = np.flatnonzero(np.isclose(xy[:, 1], surface_z, rtol=0., atol=1.e-12))
    result = []
    for position in positions:
        if np.isclose(position, surface_x, rtol=0., atol=1.e-12):
            raise ValueError('A gauge on the crack needs its sensing footprint or side')
        ids = top[(xy[top, 0] <= surface_x) & (top != plus)] if position < surface_x else top[(xy[top, 0] >= surface_x) & (top != minus)]
        ids = ids[np.argsort(xy[ids, 0])]
        if position < xy[ids[0], 0] or position > xy[ids[-1], 0]:
            raise ValueError('Gauge lies outside the modeled surface')
        result.append(float(np.interp(position, xy[ids, 0], displacement[ids, 1]))*1000)
    return result


def path_observations(directory):
    directory = Path(directory)
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    b = FractureProblem(**result['problem'])
    tracker = None
    rows = []
    tensile_volume = 0.
    shear_volume = 0.
    entries = [(directory, row, True) for row in result['history']]
    source = result.get('source')
    while source is not None:
        previous_directory = Path(source['directory'])
        previous = json.loads((previous_directory/'result.json').read_text(encoding='utf-8'))
        last_step = int(Path(source['checkpoint']).stem.rsplit('_', 1)[1])
        entries[:0] = [(previous_directory, row, False) for row in previous['history'] if row['step'] <= last_step]
        source = previous.get('source')
    for field_directory, row, emit in entries:
        with np.load(field_directory/f"fields_{row['step']:03d}.npz") as field:
            if tracker is None:
                tracker = CrackOpeningTracker(b, field['xy'], field['crack_cells'])
                triangles = field['xy'][field['triangles']]
                area = np.linalg.det(triangles[:, 1:]-triangles[:, :1])/2
                centers = triangles.mean(1)
                weights = np.polynomial.legendre.leggauss(field['crack_aperture'].shape[1])[1]/2
                plastic_j_old = np.ones(len(area))
            observed = {key: row[key] for key in ['step', 'time_s', 'face_mean_total_pressure_kpa',
                                                 'mean_face_displacement_mm']}
            observed.update(tracker.update(field['u_m']))
            # Rankine's associated potential has unit trace; the MC potential
            # has trace N_psi-1. Their multipliers distinguish tensile opening
            # strain from shear dilation without inferring a hydraulic gap.
            npsi = (1+np.sin(np.deg2rad(b.dilation_deg)))/(1-np.sin(np.deg2rad(b.dilation_deg)))
            tensile = field['bulk_tensile_multiplier']
            shear = field['bulk_shear_multiplier']*(npsi-1)
            if b.bulk_kinematics == 'finite_strain':
                # Exact plastic volume increment; split it in proportion to
                # the simultaneous logarithmic MC/Rankine volume increments.
                log_increment = tensile+shear
                factor = plastic_j_old.copy()
                np.divide(plastic_j_old*np.expm1(log_increment), log_increment,
                          out=factor, where=log_increment != 0)
                tensile_volume += float(area@(factor*tensile))*1.e6
                shear_volume += float(area@(factor*shear))*1.e6
                plastic_j_old = field['bulk_plastic_J'].copy()
                observed['bulk_plastic_volume_ml_per_m'] = float(area@(plastic_j_old-1))*1.e6
            else:
                tensile_volume += float(area@tensile)*1.e6
                shear_volume += float(area@shear)*1.e6
                observed['bulk_plastic_volume_ml_per_m'] = float(area@np.trace(field['bulk_plastic'], axis1=1, axis2=2))*1.e6
            observed['bulk_tensile_plastic_volume_ml_per_m'] = tensile_volume
            observed['bulk_shear_dilation_ml_per_m'] = shear_volume
            current_length = field['crack_current_length'] if b.interface_geometry == 'corotated' else field['crack_length_m']
            observed['crack_geometric_volume_ml_per_m'] = float((current_length[:, None]*weights*field['crack_aperture']).sum())*1.e6
            observed['crack_water_ml_per_m'] = float((field['crack_length_m'][:, None]*weights*field['crack_water']).sum())*1.e6
            active = field['bulk_tensile_multiplier'] > 0
            observed['active_tensile_min_z_m'] = float(centers[active, 1].min()) if active.any() else None
            observed['active_tensile_max_z_m'] = float(centers[active, 1].max()) if active.any() else None
            # Figure 2-15: sensor 3 is rear (-0.1 m), 2 central, 1 forward.
            for sensor, value in zip([3, 2, 1], surface_uplift(field['xy'], field['u_m'], field['crack_cells'], [-.1, 0., .1])):
                observed[f'sensor_{sensor}_uplift_mm'] = value
            observed['channel_outflow_ml_per_m_s'] = row['channel_outflow_ml_per_m_s']
            observed['connected_surface_outflow'] = observed['open_path_to_surface'] and observed['channel_outflow_ml_per_m_s'] > 0
            if emit:
                rows.append(observed)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = path_observations(args.directory)
    path = args.directory/'observations.csv'
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    events = {}
    for name, predicate in [
        ('first_surface_opening', lambda r: r['surface_endpoint_aperture_mm'] > 0),
        ('first_connected_open_path', lambda r: r['open_path_to_surface']),
        ('first_connected_surface_outflow', lambda r: r['connected_surface_outflow']),
    ]:
        events[name] = next(({'step': row['step'], 'time_s': row['time_s']} for row in rows if predicate(row)), None)
    print(json.dumps({'output': str(path), 'events': events, 'last': rows[-1]}, indent=2))


if __name__ == '__main__':
    main()
