from pathlib import Path
from typing import Any
import pandas as pd

def load_marketing_data(path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load marketing data from a csv file into a pandas Dataframe.

    Args: 
        path: File path to the raw marketing csv.
        ***read_csv_kwargs: Additional keyword arguments passed to `read.csv`.

    Returns:
        A Dataframe containing the loaded marketing data.

    Raises:
        FileNotFoundError: If no file exists at the specified path.
        ValueError: If the csv file is empty or cannot be parsed.
    
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Marketing data file not found at: {file_path}")

    try:
        return pd.read_csv(file_path, **read_csv_kwargs) #type: ignore
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Marketing data file is empty: {file_path}") from exc


    