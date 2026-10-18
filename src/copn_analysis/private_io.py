"""Keep confidential scientific inputs and generated outputs outside this checkout."""
from pathlib import Path

_CHECKOUT = Path(__file__).resolve().parents[2]

def private_input(path):
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(_CHECKOUT):
        raise ValueError("Scientific input data must remain outside the source repository.")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved

def private_output(path):
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(_CHECKOUT):
        raise ValueError("Generated data, tables, figures and models must remain outside the source repository.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

def private_directory(path):
    resolved = Path(path).expanduser().resolve()
    if resolved.is_relative_to(_CHECKOUT):
        raise ValueError("Use a private output directory outside the source repository.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
