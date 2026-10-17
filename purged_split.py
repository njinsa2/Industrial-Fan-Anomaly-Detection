from sklearn.model_selection import StratifiedKFold
import numpy as np

class PurgedStratifiedKFold:
    def __init__(self, n_splits: int = 3, purge_gap: int = 15):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=False)

    def split(self, X, y, groups=None):
        for train_idx, val_idx in self.skf.split(X, y):
            v_start, v_end = val_idx[0], val_idx[-1]
            
            #identify training samples that fall within the boundary purge window
            purge_mask = (
                (train_idx >= v_start - self.purge_gap) & 
                (train_idx <= v_end + self.purge_gap)
            )
            purged_train_idx = train_idx[~purge_mask]
            
            yield purged_train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits