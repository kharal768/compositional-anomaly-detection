# Compositional anomaly detection in multitype spatial point patterns

Code accompanying the paper "Detection of Compositional Anomalies in Multitype Spatial Point Patterns
Using Statistical Process Monitoring and Machine Learning".

## What is here
- `code/coda.py`: core functions (kNN window composition, zero replacement, ILR transform, simulation design, Phase-I-clean statistic).
- `code/ablation.py`, `code/sens.py`, `code/scen.py`, `code/scen2.py`, `code/extras.py`: multi-pattern ablation, sensitivity, comparison scenarios, robustness and step-down checks.
- `code/perloc_*.py`, `code/cozi_*.py`, `code/kulldorff.py`, `code/kd_*.py`, `code/sq_*.py`: per-location benchmarks against neighbour-preference, COZI, Squidpy and the Kulldorff scan (the per-location constructions are ours and are labelled as such in the paper).
- `code/lansing.py`, `code/realextra.py`: real-data analyses (Lansing Woods, Hyytiala, sporophores).
- `code/real_check.py`: classifier agreement check on one CODEX intestine region.
- `code/convert_lansing.py`: converts the exported Lansing Woods data to the format the scripts use.
- `annotation/`: expert-annotation protocol, blinded labelling sheet, and analysis script (not yet labelled).

## Data (not redistributed here)
- **Lansing Woods census** (`spatstat.data`, CRAN): in R, `library(spatstat.data); data(lansing)`, export as described in `code/convert_lansing.py`, then run `python code/convert_lansing.py`.
- **Hyytiala and sporophores** (`spatstat.data`): export the same way and save as `data/hyytiala.npz` and `data/sporophores.npz` with fields `x, y, m, levels`.
- **Human-intestine CODEX dataset** (Dryad, CC0): https://doi.org/10.5061/dryad.pk0p2ngrf. Extract one region (for example B004_Ascending) to `data/B004_Ascending_subset.csv`, or set the environment variable `CODEX_REGION_CSV` to its path.

## Setup
```
pip install -r requirements.txt
```
Python 3.10 or newer is recommended. The per-location scripts take minutes per pattern; several are written to run one pattern per call.

## Reproducing key results
- Power and false-alarm rates over seeds: `python code/pow_seed.py <seed>`
- Ablation (10 patterns): `python code/ablation.py`
- Lansing Woods analysis: `python code/lansing.py`
- Kulldorff scan on Lansing: `python code/kd_lansing_one.py <species_index> 999`
- COZI on Lansing: `python code/cozi_lansing.py <k> <permutations>`
- Classifier check on the CODEX region: `python code/real_check.py`

Random seeds are fixed in each script. Small differences in Monte Carlo results can occur across platforms.

## Notes
- The R analyses reported in the paper were run in R before this Python re-implementation. The Python code reproduces the Lansing Woods real-data and single-pattern power results within the reported variability.
- Expert annotations are not included; the paper states that expert validation is an open item.

## Citation
If you use this code, please cite the paper (citation to be added on publication) and the data sources above.
