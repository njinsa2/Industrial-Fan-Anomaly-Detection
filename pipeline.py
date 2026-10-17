from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from collections.abc import Callable
from plot_results import get_metrics
from purged_split import PurgedStratifiedKFold
from typing import Any
from math import log2
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sklearn
import optuna

#different sample features were tried out, pca was included initially, but was later removed since the final sample structure already had a small number of features

def get_svc_parameters(trial: optuna.Trial, **stats) -> dict[str, Any]:
    gamma_anchor = stats["gamma_base"]

    return {
        # "pca__n_components": trial.suggest_float(
        #     "pca__n_components",
        #     0.95,
        #     0.999
        # ),
        "estimator__kernel": trial.suggest_categorical(
            "estimator__kernel",
            ["poly", "rbf"]
        ),
        "estimator__C": trial.suggest_float(
            "estimator__C",
            1e-3, 100,
            log = True
        ),
        "estimator__gamma": trial.suggest_float(
            "estimator__gamma",
            gamma_anchor * 1e-2,
            gamma_anchor * 1e2,
            log = True
        )
    }

def get_mlp_parameters(trial: optuna.Trial, **stats) -> dict[str, Any]:
    return {
        # "pca__n_components": trial.suggest_float(
        #     "pca__n_components",
        #     0.95,
        #     0.999
        # ),

        "estimator__hidden_layer_sizes": trial.suggest_categorical(
            "estimator__hidden_layer_sizes",
            [
                (128,),
                (256,),
                (512,),
                (768,),
                (1024,),
                (256, 128),
                (512, 256),
                (768, 384),
                (1024, 512),
                (512, 256, 128),
            ]
        ),

        "estimator__alpha": trial.suggest_float(
            "estimator__alpha",
            1e-6,
            1e-1,
            log=True
        ),

        "estimator__learning_rate_init": trial.suggest_float(
            "estimator__learning_rate_init",
            1e-4,
            3e-2,
            log=True
        )
    }


def get_rf_parameters(trial: optuna.Trial, **stats) -> dict[str, Any]:
    return {
        # "pca__n_components": trial.suggest_float(
        #     "pca__n_components",
        #     0.95,
        #     0.999
        # ),

        "estimator__max_depth": trial.suggest_int(
            "estimator__max_depth",
            8,
            32
        ),

        "estimator__min_samples_split": trial.suggest_int(
            "estimator__min_samples_split",
            2,
            30
        ),

        "estimator__min_samples_leaf": trial.suggest_int(
            "estimator__min_samples_leaf",
            1,
            20
        ),

        "estimator__max_features": trial.suggest_categorical(
            "estimator__max_features",
            [
                "sqrt",
                "log2",
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1.0
            ]
        )
    }


def get_hgbc_parameters(trial: optuna.Trial, **stats) -> dict[str, Any]:
    return {
        # "pca__n_components": trial.suggest_float(
        #     "pca__n_components",
        #     0.95,
        #     0.999
        # ),

        "estimator__max_depth": trial.suggest_int(
            "estimator__max_depth",
            2,
            16
        ),

        "estimator__learning_rate": trial.suggest_float(
            "estimator__learning_rate",
            0.005,
            0.1,
            log=True
        ),

        "estimator__l2_regularization": trial.suggest_float(
            "estimator__l2_regularization",
            1e-6,
            1.0,
            log=True
        ),

        "estimator__min_samples_leaf": trial.suggest_int(
            "estimator__min_samples_leaf",
            2,
            30
        ),

        "estimator__max_leaf_nodes": trial.suggest_int(
            "estimator__max_leaf_nodes",
            7,
            63
        )
    }


def get_pipeline_registry()-> dict:
    #a function for the registry is implemented instead of using a constant for the purpose of ensuring no unintentionally saved states
    return {
        "SVC":(
            Pipeline([
                    ("scaler", RobustScaler()),
                    #("pca", PCA()),
                    ("estimator", SVC(class_weight = "balanced", max_iter = 30000
                                      ,probability = True, random_state = 42))
            ]),
            get_svc_parameters
        ),
        "MLPClassifier": (
            Pipeline([
                ("scaler", RobustScaler()),
                    #("pca", PCA()),
                    ("estimator", MLPClassifier(solver="adam", max_iter = 1000, random_state = 42))
            ]),
            get_mlp_parameters
        ),
        "RandomForestClassifier": (
            Pipeline([
                ("scaler", RobustScaler()),
                    #("pca", PCA()),
                    ("estimator", RandomForestClassifier(n_estimators = 300, criterion = "gini",class_weight= "balanced" , random_state = 42, n_jobs=-1))
           ]),
            get_rf_parameters
        ),
        "HistGradientBoostingClassifier": (
            Pipeline([
                ("scaler", RobustScaler()),
                    #("pca", PCA()),
                    ("estimator", HistGradientBoostingClassifier(max_iter = 200, class_weight = "balanced", early_stopping = False,random_state = 42))
            ]),
            get_hgbc_parameters
        ),
    }

def compute_data_stats(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    N, D_raw = X.shape
    
    X_scaled = RobustScaler().fit_transform(X)
    total_variance = float(np.sum(np.var(X_scaled, axis=0)))
    
    gamma_base = 1.0 / total_variance if total_variance > 0 else 1.0 / D_raw
    
    class_counts = y.value_counts()
    imbalance_ratio = float(class_counts.max() / max(class_counts.min(), 1))

    return {
        "N": N,
        "D_raw": D_raw,
        "gamma_base": gamma_base,
        "imbalance_ratio": imbalance_ratio,
    }

def bayesian_optimization(model: tuple[Pipeline, Callable], X: pd.DataFrame, y: pd.Series, groups: pd.Series = None)-> tuple[float, dict]:
    
    model_pipeline = model[0]
    get_params = model[1]
    data_stats = compute_data_stats(X = X, y = y)

    def objective(trial):
        params = get_params(trial, **data_stats)

        model = sklearn.base.clone(model_pipeline)
        model.set_params(**params)

        #3-fold purged contiguous block CV (15 files = ~2.5 min buffer gap around val folds)
        cv_splitter = PurgedStratifiedKFold(n_splits=3, purge_gap=15)
        
        scores = []

        for fold, (train_samples, val_samples) in enumerate(cv_splitter.split(X = X, y = y, groups = groups)):
            X_train = X.iloc[train_samples]
            y_train = y.iloc[train_samples]

            X_val = X.iloc[val_samples]
            y_val = y.iloc[val_samples]

            model.fit(X = X_train, y = y_train)
            y_pred = model.predict(X = X_val)

            score = f1_score(y_true = y_val, y_pred = y_pred, average = "macro")
            scores.append(score)

            intermediate_score = np.mean(scores)

            trial.report(intermediate_score, step = fold)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    #TPE sampler and median pruner are the default
    study = optuna.create_study(direction = "maximize")

    study.optimize(objective, n_trials = 35, n_jobs = 1)

    return (study.best_value, study.best_params)

def optimize_hyperparams(models: dict[str, tuple], X: pd.DataFrame, y: pd.Series, groups: pd.Series)-> tuple[dict, dict]:

    model_params = dict()
    val_scores = dict()
    for model_name in models:
        val_scores[model_name], model_params[model_name] = bayesian_optimization(models[model_name], X = X, y = y, groups = groups)

    return (model_params, val_scores)

def fit_models(models: dict[str, tuple], X: pd.DataFrame, y: pd.Series, groups: pd.Series = None)-> dict[str, float]:
    model_params, val_scores = optimize_hyperparams(models = models, X = X, y = y, groups = groups)
    print(f"Model hyperparameters dictionary: {model_params}\n")
    for model_name in models:
        models[model_name][0].set_params(**(model_params[model_name]))

        models[model_name][0].fit(X = X, y = y)

    return val_scores

def models_predict(models: dict[str, tuple], X: pd.DataFrame)-> dict[str, np.ndarray]:
    predictions = dict()
    probability_predictions = dict()
    for model_name in models:
        predictions[model_name] = models[model_name][0].predict(X)
        probability_predictions[model_name] = models[model_name][0].predict_proba(X)

    return (predictions, probability_predictions)

def save_models(models: dict[str, tuple], output_dir: str = "models") -> None:
    """Saves the fitted Pipeline object for each model in the registry as a .pkl file."""
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for model_name, (pipeline, _) in models.items():
        file_path = save_path / f"{model_name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Saved model to: {file_path}")

if __name__ == "__main__":
   #load 0dB training dataset
    train_data = pd.read_csv("data_csv/train/train_0dB.csv")
    
    train_encoded = pd.get_dummies(train_data, columns=["source_id"], dtype=float)
    X_train = train_encoded.drop(columns=["target"])
    y_train = train_encoded["target"]

    # optimize and fit models on 0dB training data
    model_registry = get_pipeline_registry()
    model_val_scores = fit_models(models=model_registry, X=X_train, y=y_train)

    print(f"Model cross-validation scores (0dB): {model_val_scores}\n")
    save_models(models=model_registry)

    #evaluate across control conditions (0dB), xleaner (6dB), and noisier (-6dB) test sets
    test_sets = {
        "0dB": "data_csv/test/test_0dB.csv",
        "6dB": "data_csv/test/test_6dB.csv",
        "-6dB": "data_csv/test/test_minus6dB.csv"
    }

    for snr_tag, test_path in test_sets.items():
        print(f"Evaluating Models on Test Dataset: {snr_tag}")
        print(f"==================================================")
        
        test_data = pd.read_csv(test_path)
        test_encoded = pd.get_dummies(test_data, columns=["source_id"], dtype=float)

        #ensure test columns align with train features
        X_test = test_encoded.reindex(columns=X_train.columns, fill_value=0.0)
        y_test = test_encoded["target"]

        model_preds, model_prob_preds = models_predict(models=model_registry, X=X_test)

        get_metrics(
            y_pred=model_preds,
            y_pred_proba=model_prob_preds,
            y_test=y_test,
            cv_scores=model_val_scores,
            dataset_tag=snr_tag
        )