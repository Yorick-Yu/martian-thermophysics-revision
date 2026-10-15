SUPPLEMENTARY SOFTWARE 1
Reanalysis for Observability of Terrain-Induced Thermal Structure in THEMIS Observations of Gasa and Kaiser

Unzip Source_Data_1.zip and this archive next to each other. Python 3.12 and NumPy are sufficient for the statistics; Matplotlib is additionally needed for diagnostic figures. The make_figures.py output is a diagnostic rendering of the archived data and does not reproduce the final author-approved figure layouts.

python reproduce_analysis.py --data ../Source_Data_1
python make_figures.py --data ../Source_Data_1 --output ./figures

The first command recomputes 10,000 paired spatial bootstrap resamples at four smoothing scales for all five 256-grid evaluation images, along with the 125-combination uncertainty envelopes. Compare analysis_recomputed.json with audit/analysis.json. It also reproduces the three Kaiser 128-grid checks in bootstrap_128_recomputed.json; compare these with audit/bootstrap_128.json. This is a reanalysis of archived projected arrays; it does not rerun the thermal simulation.

thermal_source contains the audited production entry point and column/horizon routines. These files document the actual implemented model, not a full interfacet solver. The original runner and its directly imported local modules are included. Running it requires PyTorch, the production DTM files, frozen MCD forcing and acquisition-specific configurations. These input assets and a complete thermal-regeneration environment are not bundled here. Do not treat the presence of these source files as an end-to-end reproduction claim.

Separate terrain-horizon sensitivity simulations were performed during the revision. Their analysis code, configurations and result summaries are supplied in geometry_sensitivity in the public revision repository. The statistical reproduction command above reuses archived projected arrays. Original server paths identify historical assets and must be mapped to local inputs before thermal regeneration.
