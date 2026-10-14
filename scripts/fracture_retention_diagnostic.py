"""Separate the residual-water floor from capillary water and suction traction.

The intervention removes only S_r p_f from negative-pressure traction at open
Gauss points of the top segment. It leaves the water law and accepted history
unchanged. This is a causal single-step diagnostic, not a replacement closure.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from fracture_physics import FractureProblem, cohesive_normal, interface_kinematics
from fracture_observations import surface_uplift
from pressure_dynamics import restore_dynamic_model


def load_fields(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def inventory_rows(directory, result):
    b = FractureProblem(**result['problem'])
    if b.fracture_retention != 'matrix_limit_plus_gap':
        raise ValueError('The residual-floor diagnostic requires the pre-repair retention model')
    source = result['source']
    old = load_fields(Path(source['directory'])/source['checkpoint'])
    rows = []
    for row in result['history']:
        field = load_fields(directory/f"fields_{row['step']:03d}.npz")
        order = field['crack_aperture'].shape[1]
        weights = np.polynomial.legendre.leggauss(order)[1]/2
        volume = field['crack_current_length'][-1]*weights
        old_volume = old['crack_current_length'][-1]*weights
        dt = row['step_time_s']
        pressure, width = field['crack_pressure'][-1], field['crack_aperture'][-1]
        old_pressure, old_width = old['crack_pressure'][-1], old['crack_aperture'][-1]
        residual = volume*width*b.residual_saturation*(1+b.water_compressibility_pa_inv*pressure)
        old_residual = old_volume*old_width*b.residual_saturation*(1+b.water_compressibility_pa_inv*old_pressure)
        change = field['crack_storage_change'].reshape(-1, order)[-1]
        capillary_change = change-(residual-old_residual)
        edges = field['crack_edge_flux'][-order-1:]
        along = dt*(edges[:-1]-edges[1:])
        wall = -dt*volume*field['crack_exchange'][-1]
        # Closed, empty points use a pressure reference, not an inventory balance.
        np.testing.assert_allclose(change, along+wall, atol=2.e-14, rtol=0)
        for q in range(order):
            rows.append(dict(step=row['step'], time_s=row['time_s'], point=q+1,
                dt_s=dt, aperture_um=width[q]*1.e6,
                aperture_increment_um=(width[q]-old_width[q])*1.e6,
                pressure_kpa=pressure[q]/1000,
                saturation=field['crack_saturation'][-1, q],
                residual_inventory_increment_ml_per_m=(residual[q]-old_residual[q])*1.e6,
                capillary_inventory_increment_ml_per_m=capillary_change[q]*1.e6,
                along_inflow_ml_per_m=along[q]*1.e6,
                wall_inflow_ml_per_m=wall[q]*1.e6,
                residual_wall_pressure_equivalent_kpa=-(residual[q]-old_residual[q])
                    *b.fracture_wall_resistance_pa_s_m/(2*dt*volume[q])/1000,
                residual_weighted_pressure_kpa=b.residual_saturation*pressure[q]/1000,
                capillary_weighted_pressure_kpa=(field['crack_saturation'][-1, q]
                    -b.residual_saturation)*pressure[q]/1000))
        old = field
    return rows


def top_residual_force(system, values, details):
    b, m = system.b, system.mesh
    u = values[system.local_indices][:, :, :2]*b.displacement_scale_m
    geometry = interface_kinematics(u, m.xy[system.crack_nodes], system.shape,
                                    b.interface_geometry == 'corotated')
    selected = torch.zeros_like(details['aperture'])
    selected[-1] = (details['aperture'][-1] > 0).to(values.dtype)
    delta_normal = (selected*b.residual_saturation*details['pressure'].clamp_max(0)
                    *(geometry['length']/m.crack_length)[:, None])
    delta_traction = torch.stack([delta_normal, torch.zeros_like(delta_normal)], -1)
    force = m.crack_length[:, None, None]*torch.einsum(
        'q,eqani,eqa->eni', system.crack_weights, geometry['derivative'], delta_traction)
    local = torch.cat([force, torch.zeros_like(force[:, :, :1])], -1)
    correction = torch.zeros_like(values).index_add(
        0, torch.tensor(system.local_indices.ravel()), local.flatten())
    return correction/torch.tensor(system.local_scale), delta_traction


def install_traction_intervention(system):
    if system.b.fracture_retention != 'matrix_limit_plus_gap':
        raise ValueError('The residual-floor intervention requires the pre-repair retention model')
    original = system.channel

    def channel(values, details=False):
        info = original(values, True)
        correction, delta = top_residual_force(system, values, info)
        if details:
            info['traction'] = info['traction']+delta
            return info
        return original(values)+correction

    system.channel = channel


def observations(field, row, b):
    offsets = field['crack_history_offsets']
    maximum = np.array([[field['crack_history_maximum'][left],
                         field['crack_history_maximum'][right-1]]
                        for left, right in zip(offsets[:-1], offsets[1:])])
    effective = torch.tensor(field['crack_effective_endpoints'])
    aperture = cohesive_normal(effective, torch.zeros_like(effective),
                               torch.tensor(maximum), b)[1].numpy()
    uplift = surface_uplift(field['xy'], field['u_m'], field['crack_cells'], [-.1, 0., .1])
    return dict(mean_face_displacement_mm=row['mean_face_displacement_mm'],
                sensor_3_uplift_mm=uplift[0], sensor_2_uplift_mm=uplift[1],
                sensor_1_uplift_mm=uplift[2],
                surface_aperture_um=aperture[-1, 1]*1.e6,
                surface_effective_separation_um=float(effective[-1, 1])*1.e6,
                open_path_to_surface=bool((field['crack_endpoint_edge_transmissibility'] > 0).all()),
                channel_outflow_ml_per_m_s=float(field['crack_edge_flux'][-1])*1.e6,
                top_min_pressure_kpa=float(field['crack_pressure'][-1].min())/1000,
                max_gauss_aperture_mm=float(field['crack_aperture'].max())*1000)


def check_step(model, old, row, field):
    s, b = model.system, model.system.b
    area = s.mesh.area_np
    storage = float(area@(field['bulk_water']-old['bulk_water']).mean(1))
    storage += float(s.channel_volume.numpy()@(field['crack_water']-old['crack_water']).ravel())
    swept = float(field['face_swept_volume_m2']-old['face_swept_volume_m2'])
    supplied = s.injection(s.state)
    outflow = row['outflow_ml_per_m_s']*1.e-6*b.step_time_s
    defect = (supplied-storage-swept-outflow)*1.e6
    np.testing.assert_allclose(defect, 0, atol=2.e-5)
    eigen = np.linalg.eigvalsh(field['bulk_stress'])
    nphi = (1+np.sin(np.deg2rad(b.phi_deg)))/(1-np.sin(np.deg2rad(b.phi_deg)))
    assert np.max(-eigen[:, 0]+nphi*eigen[:, -1]-2*b.effective_cohesion_pa*np.sqrt(nphi)) < 1.e-6
    assert eigen[:, -1].max() < b.tensile_strength_pa+1.e-6
    assert np.min(field['crack_maximum']-old['crack_maximum']) >= 0
    assert np.min(field['crack_damage']-old['crack_damage']) >= 0
    eps = np.finfo(float).eps
    assert np.all(field['bulk_plastic_J'] >= old['bulk_plastic_J']-64*eps*old['bulk_plastic_J'])
    for key in ('bulk_shear_multiplier', 'bulk_tensile_multiplier'):
        assert field[key].min() >= -64*eps
    for i in range(len(field['crack_cells'])):
        new_slice = slice(*field['crack_history_offsets'][i:i+2])
        old_slice = slice(*old['crack_history_offsets'][i:i+2])
        new_x, old_x = field['crack_history_fraction'][new_slice], old['crack_history_fraction'][old_slice]
        x = np.union1d(new_x, old_x)
        new_maximum = np.interp(x, new_x, field['crack_history_maximum'][new_slice])
        old_maximum = np.interp(x, old_x, old['crack_history_maximum'][old_slice])
        current = (1-x)*field['crack_effective_endpoints'][i, 0]+x*field['crack_effective_endpoints'][i, 1]
        np.testing.assert_allclose(new_maximum, np.maximum(old_maximum, current), atol=1.e-14, rtol=0)
    np.testing.assert_allclose((field['u_m']-old['u_m'])/b.step_time_s,
                               field['velocity_m_s'], atol=1.e-11)
    np.testing.assert_allclose((field['velocity_m_s']-old['velocity_m_s'])/b.step_time_s,
                               field['acceleration_m_s2'], atol=1.e-8)
    np.testing.assert_allclose(model.network.evaluate(model.network.parameters)[model.mechanical],
                               s.state[model.mechanical], atol=1.e-9, rtol=0)
    return defect


def run(directory, output, steps):
    directory, output = Path(directory), Path(output)
    result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    budget = inventory_rows(directory, result)
    with (output/'top_inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(budget[0]))
        writer.writeheader()
        writer.writerows(budget)
    record = dict(kind='single_step_residual_traction_diagnostic',
                  source=str(directory.resolve()), intervention='remove_top_open_residual_suction_only',
                  water_law='unchanged', accepted_source_history='unchanged', comparisons=[])
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        for step in steps:
            row = next(row for row in result['history'] if row['step'] == step)
            old = load_fields(directory/f'fields_{step-1:03d}.npz')
            reference = load_fields(directory/f'fields_{step:03d}.npz')
            model, _ = restore_dynamic_model(directory, f'fields_{step-1:03d}.npz', row['step_time_s'])
            s, net = model.system, model.network
            # Reconstruct the baseline from its actual predecessor before intervening.
            s.state, net.parameters = reference['state'].copy(), reference['parameters'].copy()
            baseline, reloaded = s.inspect()
            for key in ('u_m', 'crack_pressure', 'crack_water', 'bulk_stress', 'crack_traction'):
                np.testing.assert_allclose(reloaded[key], reference[key], atol=1.e-8, rtol=1.e-12)
            baseline_observed = observations(reference, row, s.b)
            s.state, net.parameters = old['state'].copy(), old['parameters'].copy()
            install_traction_intervention(s)
            print(f"Intervening at source step {step}, dt={row['step_time_s']:.9g}", flush=True)
            training = model.train(result['settings']['max_iterations'], verbose=True)
            dynamic_row, dynamic_field = model.inspect_dynamics()
            changed_row, changed = s.inspect()
            changed_row.update(training)
            changed_row.update(dynamic_row)
            changed.update(dynamic_field)
            checkpoint = f'intervention_{step:03d}.npz'
            np.savez_compressed(output/checkpoint, state=s.state, parameters=net.parameters, **changed)
            defect = check_step(model, old, changed_row, changed)
            record['comparisons'].append(dict(source_step=step, time_s=row['time_s'],
                dt_s=row['step_time_s'], baseline=baseline_observed,
                intervention=observations(changed, changed_row, s.b),
                independent_mass_defect_ml_per_m=defect, row=changed_row, checkpoint=checkpoint))
            (output/'result.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(record['comparisons'][-1]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, nargs='+', required=True)
    args = parser.parse_args()
    run(args.directory, args.output, args.steps)
