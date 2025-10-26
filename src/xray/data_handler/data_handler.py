import pandas as pd

from xray.path_manager import PathManager


class XRayDataManager:
    def __init__(self, path_manager: PathManager = None):
        if path_manager is None:
            self.path_manager = PathManager()
        else:
            self.path_manager = path_manager

    def load_data(self, file_name: str) -> pd.DataFrame:
        """Loads a single excel file into a pandas DataFrame."""
        file_path = self.path_manager.get_data_path() / file_name
        return pd.read_excel(file_path)

    def load_all_data(self) -> dict[str, pd.DataFrame]:
        """Loads all excel files in the data directory into a dictionary of pandas DataFrames."""
        data_path = self.path_manager.get_data_path()
        files = [f for f in data_path.glob("*.xlsx")]
        return {f.stem: self.load_data(f.name) for f in files}
