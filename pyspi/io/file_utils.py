import os


def file_existing_and_readable(filename):
    """
    Check if a file exists
    :param filename: Filename to check
    :return: True or False
    """
    sanitized_filename = sanitize_filename(filename)

    if os.path.exists(sanitized_filename):
        # Try to open it

        try:
            with open(sanitized_filename):
                pass

            return True
        except FileNotFoundError:
            pass
    return False


def path_exists_and_is_directory(path):
    """
    Check if a path exists and is a directory
    :param path: Path to check
    :return: True or False
    """
    sanitized_path = sanitize_filename(path, abspath=True)

    return bool(os.path.exists(sanitized_path) and os.path.isdir(path))


def sanitize_filename(filename, abspath=False):
    """
    Sanitize filename
    :param filename: name of file
    :param abspath: Get the absolute path?
    :return: sanitized filename
    """

    sanitized = os.path.expandvars(os.path.expanduser(filename))

    if abspath:
        return os.path.abspath(sanitized)

    return sanitized
