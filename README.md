# Subjectivity Mining, Assignment 5, Group 9

Ensembles of BERT, HateBERT and fBERT for offensive language detection, with hard voting,
soft voting and stacking, tested in-domain (OLID train small to OLID test) and cross-domain
(HASOC train to OLID test).

## Files
- report/A5-report-Group9.pdf: the report (LaTeX source, bibliography and the pics folder are next to it)
- A5-notebook-Group9-FINAL.ipynb: notebook with all four parts
- olid-train-small.csv, hasoc-train.csv, olid-test.csv: the datasets (0 = non-offensive, 1 = offensive)
- part1_ines_hard_voting: hard majority voting (Ines Casado Vela)
- part2_helian_soft_voting: BERT probabilities and soft voting (Helian X. Liu)
- part3_martin_stacking: HateBERT probabilities, stacking, and all ensembles on the matched pool (Martin Ranea Marina)
- part4_pablos_analysis: fBERT probabilities, Pearson correlations, McNemar tests, error analysis and the manual error categories (Pablos Tselioudis Garmendia)

## Main results (macro-F1 on the OLID test set, matched pool)
| System | In-domain | Cross-domain |
|---|---|---|
| BERT | 0.7973 | 0.7415 |
| HateBERT | 0.8018 | 0.7536 |
| fBERT | 0.8102 | 0.7966 |
| Hard voting | 0.8078 | 0.7838 |
| Soft voting | 0.8226 | 0.7984 |
| Stacking | 0.8180 | 0.8016 |

## Running the code
The packages are listed in part3_martin_stacking/requirements.txt.
- Stacking from the saved out-of-fold features (CPU): go to part3_martin_stacking and run python evaluate_stacking.py
- Training all base models again (GPU): go to part3_martin_stacking and run python a5_stacking.py
- Correlations and error analysis (CPU): go to part4_pablos_analysis and run python annotate_errors.py, then python pablos_analysis.py
