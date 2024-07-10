import pandas as pd
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def get_klines(self, *args, **kwargs) -> pd.DataFrame:
        pass
