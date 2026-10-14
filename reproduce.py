"""Recreate the prescribed pressure protocol from source and input settings."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))

from fracture_physics import FractureProblem
from fracture_pinn import run
from fracture_observations import path_observations
from pressure_dynamics import run_dynamic


def initialize(problem, settings, output, *, resume=False, smoke=False):
    return run(
        problem, output, steps=1,
        spacing=.2 if smoke else settings['spacing_m'],
        center_spacing=.3 if smoke else settings['center_spacing_m'],
        increment=.0001, dt=.001 if smoke else settings['dt_s'],
        resume=resume, learn_radii=not smoke, max_iterations=0 if smoke else 1,
        channel=True, trace_enrichment=True,
        normal_spacing=None if smoke else settings['normal_spacing_m'],
        tangent_spacing=None if smoke else settings['tangent_spacing_m'],
        local_center_spacing=None if smoke else settings['local_center_spacing_m'],
        normal_center_spacing=None if smoke else settings['normal_center_spacing_m'],
        interface_order=2 if smoke else settings['interface_order'],
        pressure_step_pa=2500., hold_time_s=.001 if smoke else 60.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ideation')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--through-segment', type=int)
    args = parser.parse_args()
    config = json.loads((ROOT / 'config/paper_case.json').read_text(encoding='utf-8'))
    problem = FractureProblem(**config['problem'])
    segments = config['segments']
    if args.through_segment is None:
        args.through_segment = len(segments)-1
    if not 0 <= args.through_segment < len(segments):
        parser.error('--through-segment must select a segment in the configured protocol')
    if args.dry_run:
        print(json.dumps({'segments': len(segments), 'final_time_s': config['final_time_s'],
                          'output': str(args.output.resolve()),
                          'initialization': config['initial_settings']}, indent=2))
        return
    if args.smoke:
        result = initialize(problem, config['initial_settings'], args.output,
                            resume=args.resume, smoke=True)
        print('Smoke status:', result['status'])
        return
    previous = None
    for i, segment in enumerate(segments[:args.through_segment + 1]):
        output = args.output / segment['name']
        resume = args.resume and (output / 'result.json').exists()
        if i == 0:
            result = initialize(problem, config['initial_settings'], output, resume=resume)
        else:
            source_result = json.loads((previous / 'result.json').read_text(encoding='utf-8'))
            result = run_dynamic(previous, source_result['history'][-1]['step'], output,
                                 segment['pressure_pa'], segment['duration_s'],
                                 segment['dt_s'], max_iterations=1, resume=resume,
                                 predict_velocity=segment.get('predict_velocity', False))
        rows = path_observations(output)
        with (output / 'observations.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Completed segment {i}: {output.name}, time={result['history'][-1]['time_s']}")
        previous = output


if __name__ == '__main__':
    main()
