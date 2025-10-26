import os
from pathlib import Path


class PathManager:
    def __init__(self, project_root: Path = None, data_path: Path = None):
        if project_root is None:
            self.project_root = Path(os.path.abspath(f'{__file__}/../..'))
        else:
            self.project_root = project_root

        if data_path is None:
            # As per your request, data sits in H:\My Drive\Lab C\X Ray\Day1
            # However, I'll also check for a local `data` directory for better portability
            local_data = self.project_root / 'data'
            gdrive_data = Path('H:/My Drive/Lab C/X Ray/Day1')
            if gdrive_data.exists() and gdrive_data.is_dir():
                self.data_path = gdrive_data
            else:
                self.data_path = local_data
        else:
            self.data_path = data_path

    def get_data_path(self) -> Path:
        return self.data_path

    def get_project_root(self) -> Path:
        return self.project_root
