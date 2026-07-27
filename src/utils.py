"""A module for assessing the quality of models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance on training and test sets.

    This function compares predictive performance to detect overfitting
    and to summarise discrimination ability.

    Args:
        model: Trained estimator with predict() and predict_proba().
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        None. Prints key metrics to stdout.
    """
    # To detect overfitting and compare the model’s behaviour across two datasets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Evaluating the model’s ability to distinguish between classes (useful in the event of an imbalance)
    print("ROC_AUC Train:", round(roc_auc_score(y_train, train_probs), 4))
    print("ROC_AUC Test:", round(roc_auc_score(y_test, test_probs), 4))

    # Detailed report shows the balance of precision/recall and F1
    print(
        "\nClassification Report Train:\n", classification_report(y_train, train_preds)
    )
    print("\nClassification Report Test:\n", classification_report(y_test, test_preds))


def plot_train_test_matrix(model, X_train, y_train, X_test, y_test, labels=None):
    """Compare confusion matrices for training and test data.

    Visual comparison helps quickly spot changes in error types
    that indicate overfitting or dataset shift.

    Args:
        model: Trained estimator with predict().
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.
        labels: Optional list of class names for display.

    Returns:
        None. Displays two confusion matrices side by side.
    """
    # Quick visual comparison of error distribution between training and test data
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Show training errors to understand what the model is learning
    ConfusionMatrixDisplay.from_estimator(
        model, X_train, y_train, display_labels=labels, cmap="Blues", ax=ax[0]
    )
    ax[0].set_title("Confusion Matrix: Train")

    # Show test errors to assess overall model quality
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=labels, cmap="Reds", ax=ax[1]
    )
    ax[1].set_title("Confusion Matrix: Test")

    plt.tight_layout()
    fig.tight_layout()
    plt.show()


def plot_curves(model, X_train, y_train, X_test, y_test):
    """Plot ROC and Precision-Recall curves for train and test sets.

    Comparing these curves highlights differences in discrimination
    and precision/recall trade-offs that may indicate overfitting
    or threshold sensitivity.

    Args:
        model: Trained estimator with predict_proba().
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        y_test: Test targets.

    Returns:
        None. Displays ROC and PR plots.
    """
    # Comparing the curves highlights differences in model behaviour on train vs test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ROC — to assess overall class discrimination ability
    RocCurveDisplay.from_estimator(model, X_train, y_train, ax=ax1, name="Train")
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax1, name="Test")
    ax1.set_title("ROC Curve: Train vs Test")

    # PR — to better understand the precision/recall trade-off at different thresholds
    PrecisionRecallDisplay.from_estimator(model, X_train, y_train, ax=ax2, name="Train")
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2, name="Test")
    ax2.set_title("PR Curve: Train vs Test")

    fig.tight_layout()
    plt.show()


def find_best_threshold(model, X_test, y_test):
    """Find the probability threshold that maximizes F1 score.

    Useful when default threshold 0.5 is suboptimal due to class imbalance
    or asymmetric costs of false positives/negatives.

    Args:
        model: Trained estimator with predict_proba().
        X_test: Test features.
        y_test: Test targets.

    Returns:
        float: Best probability threshold for positive class.
    """
    # Find the threshold that maximizes the balanced metric (F1)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Get precision/recall for all thresholds to evaluate the trade-off
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

    # Compute F1 for each precision/recall pair and select the maximum
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = thresholds[best_idx]

    # Compare the report for the default threshold and the found optimal threshold
    print("Threshold (0.5) ")
    print(classification_report(y_test, (probabilities >= 0.5).astype(int)))

    print(f"threshold ({best_threshold:.2f})")
    print(classification_report(y_test, (probabilities >= best_threshold).astype(int)))
    return best_threshold


def plot_feature_importance(model, features, top_n=10):
    """Plot and return top feature importances.

    Helps identify which features most influence model predictions
    and guides feature engineering or model interpretation.

    Args:
        model: Trained estimator with feature_importances_.
        features: List of feature names.
        top_n: Number of top features to display.

    Returns:
        pandas.DataFrame: DataFrame with features and their importances.
    """
    # Summarize feature importance to understand the model’s internal logic
    importance_df = pd.DataFrame(
        {"features": features, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="features", data=importance_df.head(top_n))
    plt.title(f"Top {top_n} important features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()
    return importance_df


def get_metrix(
    model, train_inputs, train_targets, test_inputs, test_targets, model_name, comments
):
    """Collect key metrics into a dictionary for comparison/logging.

    Enables compact storage of model performance and hyperparameters
    for experiment tracking or report generation.

    Args:
        model: Trained estimator with predict() and predict_proba().
        train_inputs: Training features.
        train_targets: Training targets.
        test_inputs: Test features.
        test_targets: Test targets.
        model_name: String name for the model.
        comments: Additional notes or comments.

    Returns:
        dict: Summary of selected metrics and model info.
    """
    # Collect key metrics into a structured format for further analysis
    return {
        "Model": model_name,
        "Hyperparams": str(model.get_params())[:150],
        "ROC_AUC Train": round(
            roc_auc_score(train_targets, model.predict_proba(train_inputs)[:, 1]), 4
        ),
        "ROC_AUC Test": round(
            roc_auc_score(test_targets, model.predict_proba(test_inputs)[:, 1]), 4
        ),
        "F1 score Train": round(
            f1_score(train_targets, model.predict(train_inputs)), 4
        ),
        "F1 score Test": round(f1_score(test_targets, model.predict(test_inputs)), 4),
        "Precision Test": round(
            precision_score(test_targets, model.predict(test_inputs)), 4
        ),
        "Recall Test": round(recall_score(test_targets, model.predict(test_inputs)), 4),
        "Comment": comments,
    }
