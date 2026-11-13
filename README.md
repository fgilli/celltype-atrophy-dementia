# Cell-type signatures of cortical atrophy across dementia etiologies

Code for my master's thesis *"Investigating Subject-Specific Cell-Type Signatures of Cortical Atrophy across Dementia Etiologies"* (M.Sc. Bioinformatics, LMU & TUM, 2026).

Subject-specific cortical thinning maps are computed with SCSR and projected onto eight cell-type maps derived from gene expression. The resulting cell-type scores are compared between diagnoses (AD, FTD and its subtypes, LBD, VAD) in three cohorts (TUM, NIFD, NACC) and used in decision-tree classification.

The repository contains no subject data.

## Scripts

The folders follow the order of the Methods section.

| Folder | Script | What it does |
|---|---|---|
| `01_cell_type_maps` | `make_cell_type_maps.R` | 8 cell-type maps on fsaverage6 from the Ecker et al. mRNA maps and the Seidlitz et al. (2020) marker genes |
| `02_resampling` | `resample_thickness.py` | FreeSurfer thickness → fsaverage6 (5 mm FWHM) → fsaverage5 |
| `03_scsr` | `train_scsr.py` | retrains SCSR on UK Biobank, one model per hemisphere |
| | `sample_scsr.py` | subject-specific deviation z-scores for a cohort |
| `04_smoothing` | `smooth_deviation_maps.py` | 5 mm smoothing of the deviation maps on fsaverage5 |
| `05_tum_preselection` | `build_tum_dataset.py` | merges the three TUM source datasets |
| | `preselect_atrophy_pattern.py` | semi-manual quality control of the TUM cohort (core regions per diagnosis) |
| | `plot_*.py` | figures for the preselection (core regions, group means, atlas) |
| `06_cell_type_scores` | `compute_cell_type_scores.py` | atrophy masses, cell-type scores, CN-referenced scores and composites |
| `07_group_analysis` | `group_mean_maps.py` | group-average deviation maps |
| | `cell_type_profiles.py` | lollipop plots of the CN-referenced scores |
| | `atrophy_heterogeneity.py` | barcode plots and within-group correlations (after Thalhammer et al. 2025) |
| | `heterogeneity_summary_plot.py` | summary figure of the within-group correlations |
| `08_classification` | `classification_pipeline.py` | decision trees with nested CV for three feature sets |
| | `compare_feature_sets_ttest.py` | corrected resampled t-test between the feature sets, Holm correction |
| | `classification_summary_plot.py` | summary figure and table values |

Every script has a short description at the top. Paths and parameters are set in the `CONFIG` / `PROFILES` block of each script.

## Data layout

Paths are relative to the repository root, results are written to `outputs/`.

```
data/
  metadata/meta_TUM.csv, meta_NIFD.csv, meta_NACC.csv    # one row per subject, same order as the arrays
                                                         # columns PTID, AGE, SEX, DX_family, Subtype (not NACC)
  deviation_maps/{tum,nifd,nacc}/zscores.fwhm5.fsavg5.{lh,rh}.npy    # (n_subjects, 10242)
  cell_maps/fsavg5_{lh,rh}_{astrocytes,endothelial,microglia,opc,oligodendrocytes,excitatory,inhibitory,pericytes}.npy
  fsaverage5/label/{lh,rh}.aparc.annot                   # from $FREESURFER_HOME/subjects/fsaverage5
  fsaverage5/surf/{lh,rh}.pial
  tum_sources/                                           # only for 05_tum_preselection
```

A few format conversions between the steps were done by hand and are not part of the scripts: the fsaverage6 cell-type maps were cut to the fsaverage5 vertices and saved as `.npy`, the resampled thickness (HDF5) was exported to the feather tables that SCSR reads, and the smoothed deviation maps (HDF5) were exported to the `.npy` arrays above.

## Running

Steps 2-4 need FreeSurfer 7.2, the raw MRI data and the UK Biobank tables for SCSR (see the script headers). The analysis itself starts at step 6:

```bash
pip install -r requirements.txt

python 06_cell_type_scores/compute_cell_type_scores.py
python 07_group_analysis/group_mean_maps.py --profile tum --group-col Subtype
python 07_group_analysis/cell_type_profiles.py --profile tum            # also nifd, nacc
python 07_group_analysis/atrophy_heterogeneity.py
python 07_group_analysis/heterogeneity_summary_plot.py
python 08_classification/classification_pipeline.py --profile tum      # also nifd, nifd_3class, nacc
python 08_classification/compare_feature_sets_ttest.py --profile tum
python 08_classification/classification_summary_plot.py
```

## References

- SCSR: Wachinger C., Hedderich D. M., Thalhammer M., Bongratz F. *Individualized mapping of aberrant cortical thickness via stochastic cortical self-reconstruction.* Medical Image Analysis 107 (2026) 103788. The code in `03_scsr` is adapted from the original SCSR code.
- Heterogeneity analysis: Thalhammer M. et al., Nat Commun 2025, https://doi.org/10.1038/s41467-025-63967-1 (code: https://github.com/Melissa1909/preterm-brain-heterogeneity)
- Marker genes: Seidlitz J. et al., Nat Commun 2020 (Supplementary Data 5). mRNA maps: Ecker et al.
- Corrected resampled t-test: Nadeau C., Bengio Y., Machine Learning 52 (2003)
