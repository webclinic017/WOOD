def CFM_metrics(y_true, y_pred):
    """
    Return the metrics used for the CFM Data Challenge 2019.

    This metrics is simply the accuracy of the prediction of positive returns
    (see below for details), but based on true *returns* (y_true) and
    probabilities (of returns being positive).

    Args
        y_true: Pandas Dataframe
            target returns. Positive returns will give True, and negative ones False.

        y_pred: Pandas Dataframe
            predicted probability of positive returns.

    """
    return ((y_true.values > 0) == (y_pred.values > 0.5)).mean()


if __name__ == '__main__':
    import pandas as pd
    CSV_FILE_1 = '--------.csv'
    CSV_FILE_2 = '--------.csv'
    df_1 = pd.read_csv(CSV_FILE_1, index_col=0, sep=',')
    df_2 = pd.read_csv(CSV_FILE_2, index_col=0, sep=',')
    print(CFM_metrics(df_1, df_2))
