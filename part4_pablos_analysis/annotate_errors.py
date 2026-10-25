"""Manual annotation of every cross-domain stacking error (39 FP, 89 FN).
Each error gets one main category, inspired by the error classes of van Aken et al. (2018, section 6).
The labels were assigned by reading each tweet. Running this file writes error_annotations_cross_domain.csv."""
import pandas as pd
from pathlib import Path

FP = {  # false positives: gold non-offensive, predicted offensive
    "Swear word in a harmless context": [83155, 23786, 64161, 14479, 84045, 80440, 16323, 43173, 88822],
    "Identity or violence terms": [30900, 39752, 72071, 11781, 24957, 93740, 46232, 79653, 60713],
    "Strong opinion without an insult": [62945, 12746, 26042, 11813, 88939, 91136, 68371, 49338, 98816, 73105, 50665],
    "Quotation or reported speech": [86385, 51052],
    "Doubtful gold label": [54053, 91217, 93101, 47834, 57869, 42404, 22812],
    "No clear cue": [31022],
}
FN = {  # false negatives: gold offensive, predicted non-offensive
    "Implicit insult without swear words": [37740, 33133, 40842, 20357, 10991, 45304, 72037, 41997, 62689, 57185, 41590, 51762],
    "Attack on a group or generalisation": [88490, 40110, 96457, 98916, 91036, 15815, 57135, 85613, 41588, 57326, 79309,
                                            27455, 12193, 11100, 22311, 56513, 55832, 96594, 98685, 31665, 38732, 79756,
                                            83416, 43782],
    "Sarcasm, irony, rhetorical question or metaphor": [44546, 22882, 34575, 59519, 10918, 60466, 57284, 79934, 46983,
                                                        58287, 63129, 65545],
    "Offence only in hashtags": [24049, 52445, 41553, 89329, 72531, 18786, 99563, 29008, 72523, 73439],
    "Swear word or slur missed": [30899, 88221, 97410, 15866, 35612],
    "Doubtful gold label": [60133, 32190, 42112, 47445, 42325, 58026, 73612, 72369, 80397, 50376, 27228, 84550, 23530,
                            71294, 83466, 14582, 22594, 56637, 61674, 55048, 76833, 16856, 25177, 72401, 14640, 78688],
}

rows = [{"id": i, "error_type": "false_positive", "category": c} for c, ids in FP.items() for i in ids]
rows += [{"id": i, "error_type": "false_negative", "category": c} for c, ids in FN.items() for i in ids]
ann = pd.DataFrame(rows)

if __name__ == "__main__":
    src = Path(__file__).resolve().parent.parent / "part3_martin_stacking/stacking_results/errors_stacking_cross_domain.csv"
    errors = pd.read_csv(src)
    assert ann.id.is_unique, "an error was labelled twice"
    assert set(ann.id) == set(errors.id), "every error must be labelled exactly once"
    merged = errors.merge(ann, on=["id", "error_type"], validate="one_to_one")
    assert len(merged) == len(errors) == 128
    merged.to_csv("error_annotations_cross_domain.csv", index=False)
    print(merged.groupby(["error_type", "category"]).size().to_string())
