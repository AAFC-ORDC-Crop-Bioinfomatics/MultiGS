ANOVA & Tukey HSD Results
    =========================

    Data source: /isilon/projects/J-002035_flaxgenomics/J-001386Frank_Lab/Flax_GS_project_J-003426/github/MultiGS/Examples/test_MultiGS-P/crossvalidation/results/cv_gs_detailed_results.csv
    Output dir : /isilon/projects/J-002035_flaxgenomics/J-001386Frank_Lab/Flax_GS_project_J-003426/github/MultiGS/Examples/test_MultiGS-P/crossvalidation/results/anova

    Factors used:
      - Models: 17 levels
      - Traits: 2 levels
      - Design : Two-way with interaction (model × trait)

    Files:
      - anova_table.csv
      - group_means.csv
      - tukey_model.csv (if >1 model)
      - tukey_trait.csv (if >1 trait)
      - tukey_model_trait_combo.csv (if >1 model and >1 trait)
      - boxplot_*.png

    Notes:
      - All replicate × fold rows are treated as independent observations.
      - Check `anova_table.csv` for significance of factors and interaction.
      - Use Tukey CSVs to see which specific groups differ.
      - Boxplots illustrate the distribution of PearsonR across groups.