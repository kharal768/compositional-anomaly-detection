# Expert annotation protocol: dominance vs even-mixing regions

## Purpose
Obtain independent expert labels for local tissue regions, to test the classifier's three classes:
background, dominance (one compartment or cell type predominates locally), and even_mixing (several
compartments present in comparable proportions locally).

## Materials
- `blinded_labelling_sheet.csv`: location_id, x_norm, y_norm, and blank label columns.
  Do NOT open or request `ANALYST_ONLY_classifier_predictions.csv` or any composition table.
- Cell-map images: for each location, the cells within the 25-nearest-neighbour window, coloured by
  compartment (epithelial, immune, stromal, other). Prepare these images from the coordinates in the
  region file. Do not show cell counts, percentages, or published neighbourhood names.

## Labels (judge visually from the cell map)
- **dominance**: one compartment visibly makes up most of the window, with the others clearly
  minor. Typical of a homogeneous patch.
- **even_mixing**: two or more compartments are visibly present in comparable amounts, with no single
  one predominating. Typical of an interface or a mixed niche.
- **background**: neither of the above is clearly visible. Use this when the window is hard to call
  or looks like typical tissue.
- **unsure**: allowed, but record a note. Unsure labels are excluded from the primary analysis and
  reported separately.

Numeric thresholds are deliberately not part of the protocol, so that the labels reflect expert
judgement. A sensitivity analysis with a stated threshold may be added later, reported separately.

## Procedure
1. Two annotators (independent) label all 625 locations. They work separately and do not discuss
   labels until both have finished.
2. Record the label in `annotator_1_label` or `annotator_2_label` and any note in the notes column.
3. Compute Cohen's kappa between annotators (script `analyse_annotations.py`).
4. Where annotators disagree, a third expert adjudicates and records `consensus_label`.
5. Annotators should complete the sheet in one sitting if possible and should not be shown the
   classifier output.

## Reporting
- Inter-annotator agreement (Cohen's kappa, with the number of 'unsure' labels).
- Classifier vs consensus: balanced accuracy, macro-F1, per-class recall, confusion matrix.
- Report the annotator count, their expertise, and the adjudication rule in the paper.
- Declare any conflicts of interest and whether annotators were blinded.
