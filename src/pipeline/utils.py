from sklearn.model_selection import StratifiedKFold
from src.models.KNN.train_knn import KnnModel
from src.models.SVM.train_svm import SvmModel
import numpy as np

def evaluate_svm_params(X, y, kernel, nu, gamma, n_splits=4):
    """
    Evaluate OneClass-per-class SVM hyperparams using stratified folds.
    Returns mean aggregate_score: conditional_accuracy * (1 - rejection_rate)
    (we want high accuracy and low rejection on known validation folds)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    rej_rates = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = np.array(y)[train_idx], np.array(y)[val_idx]

        model = SvmModel(kernel=kernel, nu=nu, gamma=gamma)
        model.train(X_tr, y_tr)
        acc = model.score(X_val, y_val)
        accs.append(acc)
        # rej_rates.append(stats['rejection_rate'])

    mean_acc = np.mean(accs)
    mean_rej = np.mean(rej_rates)
    # aggregate: prefer high acc, low rejection. weight can be tuned.
    aggregate = mean_acc * (1.0 - mean_rej)
    return aggregate, mean_acc, mean_rej
