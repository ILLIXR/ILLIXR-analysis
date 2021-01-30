from typing import List, Optional

import pandas as pd  # type: ignore
from pandas.api.types import union_categoricals  # type: ignore


def normalize_cats(
    dfs: List[pd.DataFrame],
    columns: Optional[List[str]] = None,
    include_indices: bool = True,
) -> List[pd.DataFrame]:
    if not dfs:
        return dfs

    if columns is None:
        if include_indices:
            # peel of indices into columns so that I can mutate them
            indices = [df.index.names for df in dfs]
            dfs = [df.reset_index() for df in dfs]
        columns2 = [
            column
            for column, dtype in dfs[0].dtypes.iteritems()
            if isinstance(dtype, pd.CategoricalDtype)
        ]
    else:
        columns2 = columns

    columns_union_cat = {
        column: union_categoricals(
            [df[column] for df in dfs], ignore_order=True, sort_categories=True
        ).as_ordered()
        for column in columns2
    }

    dfs = [
        df.assign(
            **{
                column: pd.Categorical(df[column], categories=union_cat.categories)
                for column, union_cat in columns_union_cat.items()
            }
        )
        for df in dfs
    ]

    if columns is None and include_indices:
        dfs = [
            df.reset_index(drop=True).set_index(index)
            for df, index in zip(dfs, indices)
        ]

    return dfs
