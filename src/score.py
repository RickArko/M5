from typing import List
import pandas as pd

from datasetsforecast.m5 import M5Evaluation

def score_submission(df: pd.DataFrame,
                     dfids: pd.DataFrame = None,
                     score_columns: List[str] = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
                    ) -> pd.DataFrame:
    """Score Submission
    """
    if dfids is None:
        dfids = get_dfids().set_index("id")

    try:
        df[score_columns]
    except KeyError:
        df = df.join(dfids)

    df_score = M5Evaluation.evaluate("data", df)
    return df_score
