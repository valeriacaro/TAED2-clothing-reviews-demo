import sys

def get_filename() -> str:
    """
            Gets file's path from user.

            Args:
                None.

            Returns:
                str containing filepath.
    """
    return str(sys.argv[1])