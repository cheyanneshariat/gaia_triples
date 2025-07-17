# Gaia Resolved Triples Catalog  
**Shariat, El-Badry, & Naoz (2025)**

---

## Overview

This repository contains the associated data and code from the Gaia Resolved Triples project (https://ui.adsabs.harvard.edu/abs/2025arXiv250616513S/abstract). This includes

1. **`triples_catalog.csv`** contains the **high-confidence triple sample**, where **each row corresponds to one triple**.
  - The data for the **inner primary**, **inner secondary**, and **tertiary** components are labeled with suffixes `1`, `2`, and `3`, respectively.
2. **`sampling_triples.ipynb`** is a notebook with code to sample a triple population with masses, separations, eccentricities, and inclinations. The code also allows sampling a complete stellar population (singles, binaries, and triples).
---

## Contents

For each star in the resolved triples, we provide:
- **Gaia DR3 columns:**
  - `source_id`
  - `parallax`
  - `pmra`, `pmdec`
  - `ra`, `dec`
  - `bp_rp` (color)
  - `phot_g_mean_mag` (G-band apparent magnitude)

- **Derived quantities:**
  - **abs_g_mag** absolute G magnitude
  - **Mass** derived from **abs_g_mag**, only applicable for main-sequence stars
  - **R_chance_score**: chance alignment probability for the triple
  - **star_type** (Main-Sequence `MS` or White Dwarf `WD`), determined based on CMD (color–magnitude diagram) location

---

## Notes

- Only **high-confidence triples** (based on low chance alignment probabilities) are included.
  - All triples, including **false matches**, are provided in `triples_all_r_chance_score.csv`.
- Components are consistently labeled, where the closest two stars are the inner binary, while the 3rd is the tertiary. Among the inner binary, the brighter (fainter) component is the primary (secondary).
- Note that using COSMIC's default `multi_dim` sampler assumes a more top-heavy primary mass function than a Kroupa IMF. We provide a corrected `multi_dim.py` file in this directory.

---
## Contact

For any questions regarding the data or code, please email: cshariat@caltech.edu.
