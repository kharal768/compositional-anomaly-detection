"""Analyse expert labels against the classifier. Run after annotators fill the sheet.
Expects: blinded_labelling_sheet.csv (annotator_1_label, annotator_2_label, consensus_label)
         ANALYST_ONLY_classifier_predictions.csv (classifier_label)
"""
import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score
sheet = pd.read_csv("blinded_labelling_sheet.csv")
pred  = pd.read_csv("ANALYST_ONLY_classifier_predictions.csv")
df = sheet.merge(pred, on="location_id")
names = ["background", "dominance", "even_mixing"]
a1 = df["annotator_1_label"].fillna("").str.strip()
a2 = df["annotator_2_label"].fillna("").str.strip()
done = (a1 != "") & (a2 != "")
print("labelled by both annotators:", int(done.sum()), "of", len(df))
if done.sum() == 0:
    print("No labels yet. Fill annotator_1_label and annotator_2_label first."); raise SystemExit
kept = done & (a1 != "unsure") & (a2 != "unsure")
print("unsure labels excluded:", int((done & ~kept).sum()))
print("Cohen's kappa (annotator 1 vs 2):", round(cohen_kappa_score(a1[kept], a2[kept]), 3))
cons = df["consensus_label"].fillna("").str.strip()
mask = kept & (cons != "") & (cons != "unsure")
if mask.sum() == 0:
    print("No consensus labels yet. Add consensus_label after adjudication."); raise SystemExit
y_true = cons[mask].map({n:i for i,n in enumerate(names)}).to_numpy()
y_pred = df.loc[mask, "classifier_label"].to_numpy()
print("evaluated locations:", int(mask.sum()))
print("balanced accuracy:", round(balanced_accuracy_score(y_true, y_pred), 3))
print("macro-F1:", round(f1_score(y_true, y_pred, average="macro"), 3))
print("per-class recall:", dict(zip(names, np.round(recall_score(y_true, y_pred, average=None, labels=[0,1,2]), 3))))
print("confusion matrix (rows = consensus, cols = classifier):")
print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0,1,2]), index=names, columns=names))
