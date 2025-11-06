import os
from pathlib import Path


class PathManager:
    def __init__(
        self,
        project_root: Path | None = None,
        data_path: Path | None = None,
        artifacts_path: Path | None = None,
    ):
        if project_root is None:
            self.project_root = Path(os.path.abspath(f"{__file__}/../.."))
        else:
            self.project_root = project_root

        if data_path is None:
            self.data_path = self.project_root / "data"
        else:
            self.data_path = data_path

        if artifacts_path is None:
            self.artifacts_path = self.project_root / "artifacts"
        else:
            self.artifacts_path = artifacts_path

    def get_data_path(self) -> Path:
        return self.data_path

    def get_artifacts_path(self) -> Path:
        return self.artifacts_path

    def get_project_root(self) -> Path:
        return self.project_root
