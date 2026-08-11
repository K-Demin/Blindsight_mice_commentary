# Analysis supplement for *Blindsight Mice or Rational Mice?*

This repository accompanies the commentary **“Blindsight Mice or Rational
Mice? Commentary on: Acute requirement for the hippocampus in putatively
conscious vision revealed by a mouse model of blindsight by Bhatla et al.
(2026)”** by Konstantin Demin, Kiyofumi Miyoshi, and Hakwan Lau.

It contains the code, methods, and reference output needed to
repeat the unequal-variance signal detection theory (SDT) analysis reported in
Figure 2 and the Supplementary Methods. The original behavioral data are not redistributed
and is available at https://iam.science/data/.

## Repository contents

- `control_median_sigma_all_experiments_plot.py` — parses the nine prepared
  behavioral tables, performs the group- and animal-level analysis, and creates
  the figure and audit tables.
- `prepare_data.py` — extracts seven tables from the official source
  archive and validates the two Figure 2 tables that must be exported from the
  legacy GraphPad file.
- `requirements.txt` — pinned third-party dependency.
- `CITATION.cff` — citation metadata for GitHub.

## Software

The package was verified with Python 3.12.13 and Pillow 12.2.0.

## Obtain the source data

The behavioral data belong to the original study and are not included here.
You can download them from the authors’ official repository:

- Data landing page: <https://iam.science/data/>
- Direct archive: <https://iam.science/data/Bhatla_et_al_2026_DATA.zip>
- Original article: Bhatla et al. (2026), *Current Biology* 36,
  2043–2062.e7, <https://doi.org/10.1016/j.cub.2026.03.031>
