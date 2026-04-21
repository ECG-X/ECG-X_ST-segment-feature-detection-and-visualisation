import os

def get_csv_paths(root_dir):
    """
    Return a list of the paths of all csv files contained in root_dir and
    any sub directories

    Args:
    - root_dir (str): The absolute path of the directory to search

    Returns:
    - A list of strings, representing the paths of all csv files found
    """
    
    csv_paths = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.csv'):
                full_path = os.path.join(dirpath, file)
                csv_paths.append(full_path)

    return csv_paths
