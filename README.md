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
- Components are consistently labeled by their role within the triple's hierarchy (inner binary + tertiary).
