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

Package verification passed all 33 included tests and the smoke run with Python 3.13.7, NumPy 2.4.4, SciPy 1.16.2, PyTorch 2.8.0 and threadpoolctl 3.6.0. These checks exercise the released files directly. The full pressure path is supplied as a regeneration command and was not rerun during packaging.

## Reproduce the pressure protocol

```text
python reproduce.py
python scripts/prepare_paper_results.py
```

`config/paper_case.json` provides the physical parameters, spatial settings and 47 successive segments of the retained pressure protocol. These are inputs, extracted from the accepted path. The first 2.5 kPa hold is quasistatic; its final state initializes the dynamic path at rest. Subsequent segments use physical-density backward Euler, from 5 to 25 kPa, and end at 540.071 s. The driver regenerates each predecessor and continues from its final accepted checkpoint. It saves full states, scalar histories and observation tables in `results/ideation/`; the table assembler creates `results/paper_data/`.

```text
python reproduce.py --resume
python reproduce.py --through-segment 4
```

The second command ends after the 5 kPa hold. A full calculation uses the manuscript's 905 output coefficients, 421 hidden radii and fine interface integration. It is substantially more expensive than the smoke example. Numerical libraries can alter accepted time subdivisions and iteration paths; check the reported force, inventory and history quantities when assessing a rerun. The package contains code and input settings; generated states and manuscript outputs are recreated by the commands above.

## Model and evidence scope

The calculation is a forward scenario with prescribed mean total face pressure and a prescribed crack route. Effective cohesion is 1 kPa. Initial suction, permeability, wall and face resistance, dilation, cohesive parameters and aperture-dependent cavity retention are assigned inputs. The liquid-bridge retention function is a modelling hypothesis. The example resolves settled holds and a finite developing interval, with distinct skeleton plastic volume, geometric cavity volume and fracture water. It does not establish a unique static capacity or reproduce an independently measured trial's face-pressure history.

The tests cover hydraulic and mechanical derivatives, moving neural tests, field prediction, finite-deformation inventory and traction, continuous opening-history transport, and history acceptance. Individual tests retain their own small-strain or finite-strain configurations; their names and inputs define their scope.
