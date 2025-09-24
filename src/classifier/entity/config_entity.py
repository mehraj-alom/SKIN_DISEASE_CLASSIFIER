from dataclasses import dataclass
from pathlib import Path


@detaclass(frozen = True)
class data_ingestion_config:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

