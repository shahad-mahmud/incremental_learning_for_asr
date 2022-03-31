import os
from typing import List
from alive_progress import alive_bar


def get_files_with_extensions(root_dir: str, extensions: List[str]) -> List[str]:
    """Iterate the root directory and sub-directories and return all files 
    with given extensions.

    Args:
        root_dir (str): The path to the root directory.
        extensions (List[str]): The list of extensions to filter.

    Returns:
        List[str]: The path of the files with given extensions.
    """
    all_files = []
    with alive_bar(spinner=None) as bar:
        for root, _, files in os.walk(root_dir):
            bar.text(root)
            for file in files:
                for extension in extensions:
                    if file.endswith(extension):
                        all_files.append(os.path.join(root, file))
                        bar()
    return all_files
