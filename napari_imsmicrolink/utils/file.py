from typing import Union, Optional, Tuple, List
from pathlib import Path
from qtpy.QtWidgets import QWidget, QFileDialog

PathLike = Union[str, Path]


def open_file_dialog(
    parent_widg: QWidget,
    single: bool = True,
    wd: PathLike = "",
    name: str = "Open files",
    file_types: str = "All Files (*)",
) -> Optional[PathLike]:
    if single is False:
        file_path, _ = QFileDialog.getOpenFileNames(
            parent_widg,
            name,
            wd,
            file_types,
        )
    else:
        file_path, _ = QFileDialog.getOpenFileName(
            parent_widg,
            name,
            wd,
            file_types,
        )

    if file_path:
        return file_path
    else:
        return None


def _generate_ims_fp_info(file_paths: Union[str, List[str]]) -> Tuple[str, str]:
    if isinstance(file_paths, list):
        fp_names = [Path(fp).name for fp in file_paths]
        fp_names = ",".join(fp_names)
        fp_names_full = [Path(fp).as_posix() for fp in file_paths]
        fp_names_full = "\n".join(fp_names_full)
    else:
        fp_names = Path(file_paths).name
        fp_names_full = Path(file_paths).as_posix()

    return fp_names, fp_names_full
