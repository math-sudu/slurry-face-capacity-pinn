"""Assemble manuscript tables from the retained, accepted pressure history."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / 'results/ideation/channel_bridge_25_filling'
OUTPUT = ROOT / 'results/paper_data'


def read_path(final=None):
    """Follow checkpoint links, excluding later rows of a shortened ancestor."""
    directory, last_step = Path(FINAL if final is None else final).resolve(), None
    segments = []
    while True:
        source_file = directory / 'result.json'
        result = json.loads(source_file.read_text(encoding='utf-8'))
        history = [r for r in result['history']
                   if last_step is None or r['step'] <= last_step]
        observations_file = directory / 'observations.csv'
        observations = {}
        if observations_file.exists():
            with observations_file.open(encoding='utf-8-sig', newline='') as f:
                observations = {int(r['step']): r for r in csv.DictReader(f)}
        rows = []
        for row in history:
            row = row.copy()
            for key, value in observations.get(row['step'], {}).items():
                if key in row or value == '':
                    continue
                row[key] = value == 'True' if value in ('True', 'False') else float(value)
            row['source_directory'] = directory.relative_to(ROOT).as_posix()
            row['checkpoint'] = f"fields_{row['step']:03d}.npz"
            if not (directory / row['checkpoint']).is_file():
                raise FileNotFoundError(directory / row['checkpoint'])
            rows.append(row)
        segments.append((source_file, observations_file, result, rows))
        source = result.get('source')
        if source is None:
            break
        last_step = int(Path(source['checkpoint']).stem.rsplit('_', 1)[1])
        # The archive records absolute Windows paths; the basename is stable
        # when this project and its results are moved together.
        name = source['directory'].replace('\\', '/').rstrip('/').rsplit('/', 1)[-1]
        directory = ROOT / 'results/ideation' / name
    segments.reverse()
    return segments


def fingerprints(segments):
    digest = hashlib.sha256()
    for result_file, observation_file, _, _ in segments:
        for path in (result_file, observation_file):
            if path.is_file():
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), digest.hexdigest()


def write_csv(name, rows, columns):
    with (OUTPUT / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    segments = read_path()
    rows = [row for _, _, _, part in segments for row in part]
    if any(b['time_s'] <= a['time_s'] for a, b in zip(rows, rows[1:])):
        raise ValueError('Accepted history times must increase across source links')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    columns = ['time_s', 'face_mean_total_pressure_kpa', 'mean_face_displacement_mm',
               'max_velocity_m_s', 'max_aperture_mm', 'mouth_aperture_mm',
               'connected_open_tip_z_m', 'bulk_plastic_volume_ml_per_m',
               'bulk_tensile_plastic_volume_ml_per_m', 'bulk_shear_dilation_ml_per_m',
               'crack_geometric_volume_ml_per_m', 'crack_water_ml_per_m',
               'sensor_3_uplift_mm', 'sensor_2_uplift_mm', 'sensor_1_uplift_mm',
               'open_path_to_surface', 'connected_surface_outflow', 'step_time_s',
               'source_directory', 'checkpoint']
    retained = [r for r in rows if r['time_s'] >= 120 - 1e-8]
    write_csv('loading_response.csv', rows, columns)
    write_csv('accepted_path.csv', retained, columns)
    stages = []
    for pressure in (7.5, 10, 12.5, 15, 17.5, 20, 22.5):
        group = [r for r in retained if abs(r['face_mean_total_pressure_kpa']-pressure) < 1e-7]
        end = group[-1]
        stages.append({
            'pressure_kpa': pressure,
            'peak_face_displacement_mm': max(r['mean_face_displacement_mm'] for r in group),
            'hold_face_displacement_mm': end['mean_face_displacement_mm'],
            'plastic_volume_ml_per_m': end['bulk_plastic_volume_ml_per_m'],
            'end_time_s': end['time_s'], 'source_directory': end['source_directory'],
            'checkpoint': end['checkpoint']})
    write_csv('pressure_holds.csv', stages, list(stages[0]))
    selected = []
    selected_times = [480, 540, 540.030, 540.038, 540.070, 540.071]
    if all(abs(retained[-1]['time_s'] - time) >= 1.e-8 for time in selected_times):
        selected_times.append(retained[-1]['time_s'])
    for time in selected_times:
        match = [r for r in retained if abs(r['time_s']-time) < 1e-8]
        if len(match) != 1:
            raise ValueError(f'Expected one accepted state at {time}, found {len(match)}')
        selected.append(match[0])
    write_csv('selected_states.csv', selected, columns)
    first, last = retained[0], retained[-1]
    increments = retained[1:]
    stored = sum(r['bulk_water_change_ml_per_m'] + r['fracture_water_change_ml_per_m'] for r in increments)
    swept = last['face_swept_volume_ml_per_m'] - first['face_swept_volume_ml_per_m']
    supplied = last['injected_volume_ml_per_m'] - first['injected_volume_ml_per_m']
    outflow = sum(r['outflow_ml_per_m_s'] * r['step_time_s'] for r in increments)
    config_hash, data_fingerprint = fingerprints(segments)
    report = {
        'assembly': 'Deterministic reading of existing accepted numerical results; no new nonlinear training.',
        'config_hash': config_hash, 'data_fingerprint': data_fingerprint,
        'segment_count': len(segments), 'accepted_rows': len(rows),
        'reported_rows': len(retained), 'pressure_holds': stages,
        'selected_states': [{key: r.get(key) for key in columns} for r in selected],
        'budget_120_to_final_ml_per_m': {'supply': supplied, 'storage': stored,
            'face_sweep': swept, 'outflow': outflow,
            'difference': supplied-stored-swept-outflow},
        'budget_role': 'Reconciliation of archived increments; independent field-based checks are recorded in B2 section 7.54.',
        'inputs': [p.relative_to(ROOT).as_posix() for a,b,_,_ in segments for p in (a,b) if p.is_file()],
        'problem': segments[-1][2]['problem'], 'settings': segments[-1][2]['settings']}
    (OUTPUT/'reported_data.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    table = [r'\begin{tabular}{rrrr}', r'\toprule',
             r'Mean face pressure & Peak displacement & Held displacement & Plastic volume \\',
             r'(kPa) & (mm) & (mm) & (mL/m) \\', r'\midrule']
    for r in stages:
        table.append(f"{r['pressure_kpa']:g} & {r['peak_face_displacement_mm']:.3f} & "
                     f"{r['hold_face_displacement_mm']:.3f} & {r['plastic_volume_ml_per_m']:.2f} " + r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}'])
    (OUTPUT/'pressure_holds.tex').write_text('\n'.join(table)+'\n', encoding='utf-8')
    print(f'Assembled {len(retained)} retained states from {len(segments)} source segments.')
    print(f"Final face displacement: {last['mean_face_displacement_mm']:.6f} mm")
    print(f'120 s to final water balance: supply={supplied:.6f}, storage={stored:.6f}, '
          f'sweep={swept:.6f}, outflow={outflow:.6f}, difference={supplied-stored-swept-outflow:.3g} mL/m')
    print('Wrote loading_response.csv, accepted_path.csv, pressure_holds.csv, selected_states.csv, reported_data.json, pressure_holds.tex')


if __name__ == '__main__':
    main()
