# Gaia Resolved Triples Catalog  
**Shariat, El-Badry, & Naoz (2025)**

---

## Overview

This repository contains the associated data and code from the Gaia Resolved Triples catalog project.

- **`triples_catalog.csv`** contains the **high-confidence triple sample**, where **each row corresponds to one triple**.
- The data for the **inner primary**, **inner secondary**, and **tertiary** components are labeled with suffixes `1`, `2`, and `3`, respectively.

---

## Contents

For each star in each triple, we provide:
- **Gaia DR3 columns:**
  - `source_id`
  - `parallax`
  - `pmra`, `pmdec`
  - `ra`, `dec`
  - `bp_rp` (color)
  - `phot_g_mean_mag` (G-band magnitude)

- **Derived quantities:**
  - **Masses** for main-sequence stars
  - **R_chance_score**: estimated **chance alignment probability** for the triple
  - **Star type** (Main-Sequence `MS` or White Dwarf `WD`), determined based on CMD (color–magnitude diagram) location

---

## Notes

- Only **high-confidence triples** (based on low chance alignment probabilities) are included.
  - All triples, including **false matches**, are provided in `triples_all_r_chance_score.csv`.
- Components are consistently labeled where the closest two stars are the inner binary while the 3rd is the tertiary. Among the inner binary, the brighter (fainter) component is the primary (secondary).
