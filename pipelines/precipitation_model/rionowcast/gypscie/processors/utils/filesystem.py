import fsspec


def make_path(path):
    """Create path in local filesystem

    Args:
        path (str): Path to create, e.g. 'data/raw'
    """
    try:
        fs = fsspec.filesystem('file')
        fs.mkdir(path)
    except:
        raise

    
def destroy_path(path, recursive=False):
    """Remove path in local filesystem

    Args:
        path (str): Path to remove, e.g. 'data/raw'
        recursive (bool, optional): If True, removes subdirectories recursively. Defaults to False.
    """
    try:
        fs = fsspec.filesystem('file')
        fs.rm(path, recursive=recursive)
    except:
        raise
