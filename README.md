# Martian thermophysics revision archive

Code and compact source data for **Observability of Terrain-Induced Thermal Structure in THEMIS Observations of Gasa and Kaiser**.

This repository supplies the revised observation-stage analysis, the audited thermal-column routines, and the code, configurations and result summaries from the terrain-horizon sensitivity calculations. The original observation-stage archive remains available at https://github.com/Yorick-Yu/martian-3d-thermophysics.

## Reproduce the reported observation statistics

Use Python 3.12 or later.

```bash
python -m pip install -r requirements.txt
python verify_release.py
python Supplementary_Software_1/reproduce_analysis.py --data Source_Data_1
python verify_release.py --check-reanalysis
```

The calculation recomputes 10,000 spatial-bootstrap resamples for five evaluation images at four smoothing scales, paired baseline comparisons, the observation-operator sensitivity envelopes, and three additional Kaiser grid checks. The recomputed JSON results are compared with the archived reference results. This workflow uses the supplied projected arrays and does not require a GPU.

## Contents

- `Supplementary_Software_1/reproduce_analysis.py` implements the corrected statistical analysis.
- `Supplementary_Software_1/thermal_source/` contains the audited production thermal entry point and its directly imported local modules.
- `Supplementary_Software_1/make_figures.py` generates diagnostic figures from the archived arrays. Their layout differs from the author-approved manuscript artwork.
- `Source_Data_1/` contains projected paired-model arrays, masks, original observation configurations, corrected statistics and compact figure source data.
- `geometry_sensitivity/analysis/` contains the comparison and terrain-preparation scripts used for the additional horizon-search calculations.
- `geometry_sensitivity/configs/` and `geometry_sensitivity/results/` retain the available small JSON configurations and result summaries, including intermediate versions. Later files do not erase earlier findings.
- `SHA256SUMS.json` records the release file hashes.

## Thermal simulations and input access

The thermal model calculates independent vertical columns with local-slope illumination and terrain-horizon shadowing. Thermal regeneration uses PyTorch and also requires the production terrain inputs, acquisition-specific frozen Mars Climate Database forcing and, for continuation runs, saved thermal states. The original HiRISE and THEMIS products are available through the NASA Planetary Data System at https://pds.nasa.gov. Product identifiers are retained in the supplied configurations and manuscript.

The full thermal-state archive and associated frozen forcing inputs are large and remain in the project archive. They are available from the corresponding author on reasonable request, as specified in the manuscript. Historical absolute paths in archived configurations identify original inputs and outputs. They must be mapped to the corresponding local assets before running thermal or geometry scripts. These archived scripts are not a portable one-command thermal regeneration workflow.

The sensitivity summaries report both distance and azimuth effects. They retain the observed Gasa distance sensitivity and incomplete valid-terrain coverage. No claim of convergence outside the available terrain is made.

## Version

The manuscript release is tagged `v1.0.0`. Use the tag or its commit when citing this code. The `SHA256SUMS.json` manifest and `verify_release.py` allow the released files and observation-stage results to be checked independently.
