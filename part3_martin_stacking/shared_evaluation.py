import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,ConfusionMatrixDisplay

def evaluate_predictions(y_true, y_pred, model_name, setting, show_matrix=True):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0,1], zero_division=0
    )

    summary = pd.DataFrame([{
        "model": model_name,
        "setting": setting,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "precision_0": p[0], "recall_0": r[0], "f1_0": f1[0], "support_0": support[0],
        "precision_1": p[1], "recall_1": r[1], "f1_1": f1[1], "support_1": support[1],
    }])

    if show_matrix:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(
            cm, display_labels=["Non-offensive (0)", "Offensive (1)"]
        )
        disp.plot(values_format="d")
        plt.title(f"{model_name} — {setting}")
        plt.tight_layout()
        plt.show()

    return summary