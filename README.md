# Conservative neural virtual work for unsaturated face-pressure response

Source code for *Conservative hydraulic constraints for neural virtual-work simulation of unsaturated face-pressure response*.

The solver represents displacement with trainable Gaussian features, discontinuity enrichment and local traces. A mixed hydraulic layer solves matrix and fracture water balances for every trial displacement. Feature derivatives include implicit pressure feedback and the variation of the neural virtual-work tests. Plastic, cohesive, water and velocity histories are committed together.

## Install and verify

Use Python 3.11 or later in a virtual environment:

```text
python -m pip install -r requirements.txt
python -m pytest tests -q
python reproduce.py --dry-run
python reproduce.py --smoke --output results/smoke
```

The smoke example uses a coarse integration mesh and one small pressure-controlled increment. It checks the packaged imports, hydraulic solve, neural output equilibration and checkpoint production. It is separate from the manuscript's numerical resolution.

## Reproduce the pressure protocol

```text
python reproduce.py
python scripts/prepare_paper_results.py
```

`config/paper_case.json` provides the physical parameters, spatial settings and 49 successive segments of the retained pressure protocol. The first 2.5 kPa hold is quasistatic; its final state initializes the dynamic path at rest. Subsequent segments use physical-density backward Euler, from 5 to 25 kPa, and end at 540.0724375 s. The last segment uses a constant-velocity displacement predictor before output equilibration. The driver regenerates each predecessor and continues from its final accepted checkpoint. It saves full states, scalar histories and observation tables in `results/ideation/`; the table assembler creates `results/paper_data/`.

```text
python reproduce.py --resume
python reproduce.py --through-segment 4
```

The second command ends after the 5 kPa hold. A full calculation uses the manuscript's 905 output coefficients, 421 hidden radii and fine interface integration. It is substantially more expensive than the smoke example. Numerical libraries can alter accepted time subdivisions and iteration paths; check the reported force, inventory and history quantities when assessing a rerun. The package contains code and input settings; generated states and manuscript outputs are recreated by the commands above.

## Model and evidence scope

The calculation is a forward scenario with prescribed mean total face pressure and a prescribed crack route. Effective cohesion is 1 kPa. Initial suction, permeability, wall and face resistance, dilation, cohesive parameters and aperture-dependent cavity retention are assigned inputs. The liquid-bridge retention function is a modelling hypothesis. The example resolves settled holds and a finite developing interval, with distinct skeleton plastic volume, geometric cavity volume and fracture water.

## Recover the mechanism and adaptation data

After the full protocol, the following commands generate the signed water partition, local transport and filling histories, plastic fields, and neural-feature history:

```text
python scripts/export_mechanism_history.py
python scripts/export_crack_transport.py
python scripts/export_crack_transition.py
python scripts/export_plastic_fields.py
python scripts/export_neural_adaptation.py --history-only
```

To reconstruct the fixed-radius output solution in the two increments selected for the adaptation comparison, run:

```text
python scripts/export_neural_adaptation.py
```

These two reconstructions reuse each increment's preceding physical state, time step and displacement predictor. Their fields are written to `results/paper_data/feature_comparison.npz`, and scalar comparisons to `results/paper_data/feature_comparison.csv`.

The tests cover hydraulic and mechanical derivatives, moving neural tests, field prediction, finite-deformation inventory and traction, continuous opening-history transport, and history acceptance. Individual tests retain their own small-strain or finite-strain configurations; their names and inputs define their scope.
