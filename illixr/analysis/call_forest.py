import pandas as pd  # type: ignore


class CallForest:
    _data: pd.DataFrame

    def __init__(self, data: pd.DataFrame) -> None:
        self._data = data
