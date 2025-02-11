# -*- coding: utf-8 -*-

from prefeitura_rio.pipelines_utils.logging import log  # pylint: disable=E0611, E0401


def print_error(
    message: str = "ERROR",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in red."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            log(f"{string_before}\x1b[1;30;41m[ {message} ]\x1b[0m{string_after}")
        else:
            log(f"{string_before}\x1b[31m{message}\x1b[0m{string_after}")


def print_warning(
    message: str = "WARNING",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in yellow."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            log(f"{string_before}\x1b[1;30;43m[ {message} ]\x1b[0m{string_after}")
        else:
            log(f"{string_before}\x1b[33m{message}\x1b[0m{string_after}")


def print_ok(
    message: str = "OK",
    verbose: bool = True,
    skip_line_before: bool = True,
    skip_line_after: bool = True,
    bold: bool = False,
) -> None:
    """Print message in green."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        if bold:
            log(f"{string_before}\x1b[1;30;42m[ {message} ]\x1b[0m{string_after}")
        else:
            log(f"{string_before}\x1b[32m{message}\x1b[0m{string_after}")


def print_info(
    message: str,
    verbose: bool = True,
    skip_line_before: bool = False,
    skip_line_after: bool = False,
) -> None:
    """Print info."""
    if verbose:
        string_before = "\n" if skip_line_before else ""
        string_after = "\n" if skip_line_after else ""
        log(f"{string_before}{message}{string_after}")
