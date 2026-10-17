import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



def create_results_directories(base_path: Path, dataset_tag: str)-> dict[str, Path]:
    #create directory hierarchy 
    tag_base = base_path / dataset_tag
    subfolders = {
        "cm": tag_base / "confusion_matrices",
        "roc": tag_base / "roc_curves",
        "accuracy": tag_base / "accuracy",
        "cv_scores": tag_base / "cv_scores",
        "macro_f1": tag_base / "macro_f1",
        "roc_auc": tag_base / "roc_auc",
        "gen_gap": tag_base / "generalization_gap",
        "abnormal": tag_base / "abnormal_class_metrics",
    }
    
    for folder in subfolders.values():
        folder.mkdir(parents=True, exist_ok=True)
        
    return subfolders


def plot_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, save_path: Path, color: str = "skyblue")-> None:
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=df, x=x_col, y=y_col, palette=[color] * len(df))
    
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Model Name", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.xticks(rotation=15)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.3f}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center", 
            va="bottom",
            fontsize=10, 
            xytext=(0, 3), 
            textcoords="offset points"
        )
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_metrics(y_pred: dict[str, np.ndarray], y_pred_proba: dict[str, np.ndarray], y_test: np.ndarray, cv_scores: dict[str, float],dataset_tag: str = "0dB")-> None:
    results_base = Path("results")
    dirs = create_results_directories(results_base, dataset_tag)

    full_reports = {}
    confusion_matrices = {}
    generalization_gaps = {}
    roc_auc_scores = {}
    summary_data = []

    plt.figure(figsize=(9, 6))

    for model_name in y_pred:
        preds = y_pred[model_name]
        probs = y_pred_proba[model_name][:, 1]
        
        report_dict = classification_report(y_true=y_test, y_pred=preds, output_dict=True)
        report_str = classification_report(y_true=y_test, y_pred=preds)
        
        full_reports[model_name] = report_dict
        cm = confusion_matrix(y_true=y_test, y_pred=preds)
        confusion_matrices[model_name] = cm
        
        test_macro_f1 = report_dict["macro avg"]["f1-score"]
        gen_gap = cv_scores[model_name] - test_macro_f1
        generalization_gaps[model_name] = gen_gap

        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc_val = auc(fpr, tpr)
        roc_auc_scores[model_name] = roc_auc_val

        print(f"=== {model_name} [{dataset_tag}] ===")
        print(report_str)
        print(f"Generalization Gap: {gen_gap:.5f}\n" + "-"*50)

        #confusion matrix heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Normal (0)", "Abnormal (1)"],
            yticklabels=["Normal (0)", "Abnormal (1)"]
        )
        plt.xlabel("Predicted Label", fontsize=11)
        plt.ylabel("True Label", fontsize=11)
        plt.title(f"{model_name} ({dataset_tag}) - Confusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(dirs["cm"] / f"cm_{model_name}.png", dpi=300)
        plt.close()

        #individual roc curves
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc_val:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Chance (AUC = 0.50)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (FPR)", fontsize=11)
        plt.ylabel("True Positive Rate (TPR)", fontsize=11)
        plt.title(f"{model_name} ({dataset_tag}) - ROC Curve", fontsize=13, fontweight="bold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(dirs["roc"] / f"roc_{model_name}.png", dpi=300)
        plt.close()

        plt.figure(1)
        plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc_val:.3f})")

        abnormal_key = "1" if "1" in report_dict else 1
        summary_data.append({
            "Model": model_name,
            "Accuracy": report_dict["accuracy"],
            "CV Score": cv_scores[model_name],
            "Macro F1": test_macro_f1,
            "ROC-AUC": roc_auc_val,
            "Generalization Gap": gen_gap,
            "Abnormal Precision": report_dict[abnormal_key]["precision"],
            "Abnormal Recall": report_dict[abnormal_key]["recall"],
            "Abnormal F1": report_dict[abnormal_key]["f1-score"],
        })

    #combined roc plot
    plt.figure(1)
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR)", fontsize=11)
    plt.title(f"ROC Curve Comparison ({dataset_tag}) - All Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(dirs["roc"] / "roc_all_models.png", dpi=300)
    plt.close()

    df_metrics = pd.DataFrame(summary_data)

    #single-metric bar plots
    plot_bar_chart(df_metrics, "Model", "Accuracy", f"Accuracy ({dataset_tag})", "Accuracy", dirs["accuracy"] / "accuracy_bar_plot.png", "skyblue")
    plot_bar_chart(df_metrics, "Model", "CV Score", f"CV Scores ({dataset_tag})", "CV Score (Macro F1)", dirs["cv_scores"] / "cv_scores_bar_plot.png", "mediumseagreen")
    plot_bar_chart(df_metrics, "Model", "Macro F1", f"Test Macro F1 ({dataset_tag})", "Macro F1 Score", dirs["macro_f1"] / "macro_f1_bar_plot.png", "coral")
    plot_bar_chart(df_metrics, "Model", "ROC-AUC", f"ROC-AUC ({dataset_tag})", "ROC-AUC Score", dirs["roc_auc"] / "roc_auc_bar_plot.png", "orchid")
    plot_bar_chart(df_metrics, "Model", "Generalization Gap", f"Generalization Gap ({dataset_tag})", "CV Score - Test F1", dirs["gen_gap"] / "generalization_gap_bar_plot.png", "goldenrod")

    #grouped bar plot for abnormal class
    df_abnormal = df_metrics.melt(
        id_vars=["Model"], 
        value_vars=["Abnormal Precision", "Abnormal Recall", "Abnormal F1"],
        var_name="Metric", 
        value_name="Score"
    )

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_abnormal, x="Model", y="Score", hue="Metric", palette="Blues_d")
    plt.title(f"Abnormal Class Metrics ({dataset_tag})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Model Name", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0, 1.15)
    plt.legend(loc="lower right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center", 
                va="bottom",
                fontsize=9, 
                xytext=(0, 2), 
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(dirs["abnormal"] / "abnormal_class_metrics_bar_plot.png", dpi=300)
    plt.close()

    print(f"Saved evaluation results to 'results/{dataset_tag}/'\n")