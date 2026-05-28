import numpy as np


class TestCaseGenerator:
    def __init__(self, sel_method, X_test, y_test, y_pred, y_pos_proba, thr_pos=0.5, sample=None):
        """
        :param thr_pos: The threshold for y_pos_proba to be considered as positive (default: 0.5)
        """
        if sel_method == 'border':
            cond = (y_pos_proba >= thr_pos - 0.05) & (y_pos_proba <= thr_pos + 0.05)
            # y_target = (y_pred == 0).astype(int)  # equivalent to 1 - y_pred
            y_target = 1 - y_pred
        elif sel_method == 'neg_border':
            cond = (y_pos_proba >= thr_pos - 0.05) & (y_pred == 0)
            # y_target = np.ones(len(y_test), dtype=int)
            y_target = 1 - y_pred  # filtered later
        elif sel_method == 'pos_border':
            cond = (y_pos_proba <= thr_pos + 0.05) & (y_pred == 1)
            # y_target = np.zeros(len(y_test), dtype=int)
            y_target = 1 - y_pred  # filtered later
        elif sel_method == 'auto-refuse':
            cond = (y_pos_proba < thr_pos)
            y_target = 1 - y_pred

            if sample:
                # select the percentile-based threshold, between 25, 50 and 75 percentile, that yields the closest to 'sample' but larger number of records
                percentiles = [25, 50, 75]
                best_cond = cond
                best_diff = float('inf')
                for percentile in percentiles:
                    thr_pos_min = np.percentile(y_pos_proba[y_pos_proba < thr_pos], percentile)
                    cond2 = y_pos_proba > thr_pos_min
                    cond3 = cond & cond2
                    diff = abs(cond3.sum() - sample)
                    if diff < best_diff and cond3.sum() >= sample:
                        best_cond = cond3
                        best_diff = diff
                cond = best_cond

        elif sel_method == 'fp':
            cond = (y_test == 0) & (y_pred == 1)
            y_target = np.zeros(len(y_test), dtype=int)
        elif sel_method == 'fn':
            cond = (y_test == 1) & (y_pred == 0)
            y_target = np.zeros(len(y_test), dtype=int)
        else:
            raise ValueError("Method must be in ['border', 'neg_border', 'pos_border', 'auto-refuse', 'fp', 'fn']")

        # TODO: insert other selection strategies that consider for instance some fairness-relevant record categories
        #  (e.g., women, young, black people, etc.)

        self.X = X_test[cond]
        self.y = y_test[cond]
        self.y_pred = y_pred[cond]
        self.y_pos_proba = y_pos_proba[cond]
        self.y_target = y_target[cond]

        if sample:
            self.X = self.X.iloc[:sample]
            self.y = self.y.iloc[:sample]
            self.y_pred = self.y_pred[:sample]
            self.y_pos_proba = self.y_pos_proba[:sample]
            self.y_target = self.y_target[:sample]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X.iloc[idx],
            self.y.iloc[idx].item(),  # convert from numpy scalar to Python scalar
            self.y_pred[idx].item(),
            self.y_pos_proba[idx].item(),
            self.y_target[idx].item()
        )
