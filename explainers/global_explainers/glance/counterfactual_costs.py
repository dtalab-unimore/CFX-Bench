from typing import Callable, List, Dict
import numpy as np
import pandas as pd


def build_dist_func_dataframe(
    X: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    n_bins: int = 10,
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.Series]:
    feat_intervals = {
        col: ((max(X[col]) - min(X[col])) / n_bins) for col in numerical_columns
    }

    def bin_numericals(instances: pd.DataFrame):
        ret = instances.copy()
        for col in numerical_columns:
            ret[col] /= feat_intervals[col]
        return ret
    
    def dist_f(X1: pd.DataFrame, X2: pd.DataFrame) -> pd.Series:
        X1 = bin_numericals(X1)
        X2 = bin_numericals(X2)
        
        ret = (X1[numerical_columns] - X2[numerical_columns]).abs().sum(axis="columns")
        ret += (X1[categorical_columns] != X2[categorical_columns]).astype(int).sum(axis="columns")

        return ret
    
    return dist_f

def build_dist_func_dataframe_work(
    X: pd.DataFrame,
    numerical_columns: List[str],
    categorical_columns: List[str],
    n_bins: int = 10,
) -> Callable[[pd.DataFrame, pd.DataFrame], pd.Series]:
    feat_intervals = {
        col: ((max(X[col]) - min(X[col])) / n_bins) for col in numerical_columns
    }

    def bin_numericals(instances: pd.DataFrame):
        ret = instances.copy()
        for col in numerical_columns:
            ret[col] /= feat_intervals[col]
        return ret
    
    def dist_f(X1: pd.DataFrame, X2: pd.DataFrame) -> pd.Series:
        X1 = bin_numericals(X1)
        X2 = bin_numericals(X2)
        
        ret = (X1[numerical_columns] - X2[numerical_columns]).abs().sum(axis="columns")
        costs = []
        i=0
# 
        for (index1, row1), (index2, row2) in zip(X1.iterrows(), X2.iterrows()):
            cost = 0
            for col in categorical_columns:
                if col == 'skills':
                     # Check if column contains lists
                    list1 = row1[col].split(',')
                    list2 = row2[col].split(',')
                    list_cost = sum(1 for item1 in list1 if item1 not in list2)
                    cost += list_cost
                else:
                    if (row1[col] != row2[col]):
                        cost += 1

            costs.append(cost)

        return ret + pd.Series(costs)
    
    return dist_f