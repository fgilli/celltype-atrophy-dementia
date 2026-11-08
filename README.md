# Subject-Specific Cell-Type Signatures of Cortical Atrophy across Dementia Etiologies

Code of the master's thesis *"Investigating Subject-Specific Cell-Type Signatures of Cortical Atrophy across Dementia Etiologies"* (M.Sc. Bioinformatics, LMU & TUM, 2026).

Subject-specific cortical thinning maps are computed with Stochastic Cortical Self-Reconstruction (SCSR) and projected onto eight cell-type maps derived from gene expression. The resulting cell-type profiles are compared across diagnoses (AD, FTD and its subtypes, LBD, VAD) in three cohorts (TUM, NIFD, NACC) and used for interpretable decision-tree classification.

No subject data is included in this repository.

## Workflow

The folders follow the order of the Materials and Methods section.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_cell_type_maps/make_cell_type_maps.R` | Builds 8 cell-type maps on fsaverage6 from the Ecker et al. mRNA maps and the Seidlitz et al. (2020) marker gene sets (marker genes z-scored across vertices, then averaged). |
| 2 | `02_resampling/resample_thickness.py` | Resamples FreeSurfer 7.2 cortical thickness to fsaverage6 with 5 mm FWHM smoothing (`mri_surf2surf`) and keeps the fsaverage5 vertices. |
| 3 | `03_scsr/train_scsr.py` | Retrains the SCSR model on UK Biobank (one model per hemisphere). |
| 3 | `03_scsr/sample_scsr.py` | Computes subject-specific deviation z-scores for every subject of a cohort. |
| 4 | `04_smoothing/smooth_deviation_maps.py` | Smooths the deviation maps on fsaverage5 (5 mm FWHM, medial wall and corpus callosum excluded). |
| 5 | `05_tum_preselection/build_tum_dataset.py` | Combines the three TUM source datasets into one cohort. |
| 5 | `05_tum_preselection/preselect_atrophy_pattern.py` | Semi-manual preselection of the TUM cohort: flags subjects whose atrophy pattern does not match the core ROI of their diagnosis, for manual review. |
| 5 | `05_tum_preselection/plot_*.py` | Optional illustrations for the preselection: core ROIs, group means with the ROI outlined, and an atlas reference. |
| 6 | `06_cell_type_scores/compute_cell_type_scores.py` | Atrophy weights, total, regional (8 groups) and parcel (34 DK) atrophy masses, raw and normalized cell-type scores, and CN-referenced z-scores with glial, vascular and neuronal composites. |
| 7 | `07_group_analysis/group_mean_maps.py` | Group-average deviation maps (thinning only). |
| 7 | `07_group_analysis/cell_type_profiles.py` | CN-referenced lollipop plots (cell types and composites, by family and subtype) and the ±1 SD in-band tables. |
| 7 | `07_group_analysis/atrophy_heterogeneity.py` | Barcode plots and within-group Spearman correlations of extranormal parcels, following Thalhammer et al. (2025). |
| 7 | `07_group_analysis/heterogeneity_summary_plot.py` | Summary figure of the within-group correlations. |
| 8 | `08_classification/classification_pipeline.py` | Decision trees for healthy vs. sick and bvFTD vs. SD, using three feature sets (regional atrophy, cell type, combined). Evaluated with nested CV (5 outer folds × 10 repeats) plus a final tree for interpretation. |
| 8 | `08_classification/compare_feature_sets_ttest.py` | Corrected repeated k-fold t-test (Nadeau & Bengio, 2003) of each feature set against regional atrophy, with Holm correction. |
| 8 | `08_classification/classification_summary_plot.py` | Summary figure and table values of the classification. |

Each script starts with a description of its inputs, method and outputs. Settings such as paths, FWHM, grids and seeds are set in a `CONFIG` / `PROFILES` block at the top of each script.

## Expected data layout

All paths are relative to the repository root, and results are written to `outputs/`.

```
data/
  metadata/meta_TUM.csv, meta_NIFD.csv, meta_NACC.csv   # one row per subject, row-aligned with the arrays;
                                                        # columns PTID, AGE, SEX, DX_family, Subtype (not NACC)
  deviation_maps/{tum,nifd,nacc}/zscores.fwhm5.fsavg5.{lh,rh}.npy   # (n_subjects, 10242)
  cell_maps/fsavg5_{lh,rh}_{astrocytes,endothelial,microglia,opc,oligodendrocytes,excitatory,inhibitory,pericytes}.npy
  fsaverage5/label/{lh,rh}.aparc.annot                  # from $FREESURFER_HOME/subjects/fsaverage5
  fsaverage5/surf/{lh,rh}.pial
  tum_sources/                                          # only for step 5 (see build_tum_dataset.py)
```

Some intermediate conversions were done outside these scripts and are not included:

- **Cell-type maps:** converted from fsaverage6 to the fsaverage5 (ico5) vertex set.
- **Resampled thickness:** the HDF5 files from step 2 were exported to the feather tables that SCSR reads.
- **Smoothed deviation maps:** the HDF5 files from step 4 were exported to the `.npy` arrays above.

## Running

```bash
pip install -r requirements.txt

# steps 2-4 need FreeSurfer 7.2 (FREESURFER_HOME) and the raw MRI data; see the script headers
# 03_scsr/model/ must contain the network definition named in training_configs/training_config.yaml

python 06_cell_type_scores/compute_cell_type_scores.py
python 07_group_analysis/group_mean_maps.py --profile tum --group-col Subtype
python 07_group_analysis/cell_type_profiles.py --profile tum        # likewise nifd, nacc
python 07_group_analysis/atrophy_heterogeneity.py
python 07_group_analysis/heterogeneity_summary_plot.py
python 08_classification/classification_pipeline.py --profile tum  # likewise nifd, nifd_3class, nacc
python 08_classification/compare_feature_sets_ttest.py --profile tum
python 08_classification/classification_summary_plot.py
```

## References and acknowledgements

- SCSR: Wachinger C., Hedderich D. M., Thalhammer M., Bongratz F. *Individualized mapping of aberrant cortical thickness via stochastic cortical self-reconstruction.* Medical Image Analysis 107 (2026) 103788. `03_scsr` is adapted from the original SCSR code.
- Heterogeneity analysis: Thalhammer M. et al., Nat Commun 2025, https://doi.org/10.1038/s41467-025-63967-1 (code: https://github.com/Melissa1909/preterm-brain-heterogeneity).
- Cell-type marker genes: Seidlitz J. et al., Nat Commun 2020 (Supplementary Data 5). Vertex-wise mRNA maps: Ecker et al.
- Corrected resampled t-test: Nadeau C., Bengio Y., Machine Learning 52 (2003).
