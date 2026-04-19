import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(command: list[str], cwd: Path | None = None):
    cmd_str = " ".join(str(part) for part in command)
    print(f"> {cmd_str}")
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {cmd_str}\n"
            f"{(result.stderr or '').strip()[-2000:]}"
        )
    return result


def get_venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def ensure_pip_available(python_executable: str, cwd: Path):
    try:
        run_cmd([python_executable, "-m", "pip", "--version"], cwd=cwd)
        return
    except RuntimeError:
        print("pip is not available. Trying ensurepip...")

    run_cmd([python_executable, "-m", "ensurepip", "--upgrade"], cwd=cwd)


def ensure_virtualenv_available(python_executable: str, cwd: Path):
    try:
        run_cmd([python_executable, "-m", "virtualenv", "--version"], cwd=cwd)
        return
    except RuntimeError:
        print("virtualenv is not available. Installing it...")

    ensure_pip_available(python_executable, cwd)

    install_attempts = [
        [python_executable, "-m", "pip", "install", "-U", "virtualenv"],
        [python_executable, "-m", "pip", "install", "--break-system-packages", "-U", "virtualenv"],
        [python_executable, "-m", "pip", "install", "--user", "-U", "virtualenv"],
    ]

    last_error = None
    for install_cmd in install_attempts:
        try:
            run_cmd(install_cmd, cwd=cwd)
            run_cmd([python_executable, "-m", "virtualenv", "--version"], cwd=cwd)
            return
        except RuntimeError as error:
            last_error = error

    raise RuntimeError(
        "Unable to install virtualenv automatically. "
        "Please install it manually then re-run this script."
    ) from last_error


def create_virtual_environment(python_executable: str, venv_path: Path, backend: str, cwd: Path):
    selected_backend = backend.lower().strip()
    if selected_backend not in {"auto", "venv", "virtualenv"}:
        raise ValueError("Invalid venv backend. Expected one of: auto, venv, virtualenv")

    if selected_backend in {"auto", "venv"}:
        try:
            run_cmd([python_executable, "-m", "venv", str(venv_path)], cwd=cwd)
            return
        except RuntimeError as venv_error:
            if selected_backend == "venv":
                raise
            print("Built-in venv failed; falling back to virtualenv...")
            print(str(venv_error))

    ensure_virtualenv_available(python_executable, cwd)
    run_cmd(
        [python_executable, "-m", "virtualenv", "--python", python_executable, str(venv_path)],
        cwd=cwd,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update virtualenv and install project requirements."
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory (where requirements.txt lives).",
    )
    parser.add_argument(
        "--venv-path",
        default=".venv",
        help="Virtualenv path (absolute or relative to project dir).",
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
        help="Requirements file path (absolute or relative to project dir).",
    )
    parser.add_argument(
        "--python",
        default="",
        help="Python interpreter used to create the virtualenv (default: current interpreter).",
    )
    parser.add_argument(
        "--venv-backend",
        default="auto",
        choices=["auto", "venv", "virtualenv"],
        help="Environment creation backend. 'auto' tries venv first then falls back to virtualenv.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete virtualenv first then recreate it.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Create virtualenv only and skip pip install -r requirements.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_dir = Path(args.project_dir).resolve()
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    venv_path = Path(args.venv_path)
    if not venv_path.is_absolute():
        venv_path = (project_dir / venv_path).resolve()

    requirements_path = Path(args.requirements)
    if not requirements_path.is_absolute():
        requirements_path = (project_dir / requirements_path).resolve()

    python_executable = (args.python or sys.executable).strip()
    if not python_executable:
        raise RuntimeError("No Python interpreter selected for virtualenv creation.")

    print(f"Project directory: {project_dir}")
    print(f"Virtualenv path : {venv_path}")
    print(f"Requirements    : {requirements_path}")
    print(f"Python          : {python_executable}")
    print(f"Venv backend    : {args.venv_backend}")

    if args.recreate and venv_path.exists():
        print(f"Removing existing virtualenv: {venv_path}")
        shutil.rmtree(venv_path)

    if not venv_path.exists():
        create_virtual_environment(
            python_executable=python_executable,
            venv_path=venv_path,
            backend=args.venv_backend,
            cwd=project_dir,
        )
    else:
        print("Virtualenv already exists. Reusing it.")

    venv_python = get_venv_python(venv_path)
    if not venv_python.exists():
        raise FileNotFoundError(f"Virtualenv python not found: {venv_python}")

    run_cmd([str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], cwd=project_dir)

    if not args.skip_install:
        if not requirements_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
        run_cmd([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)], cwd=project_dir)
    else:
        print("Skipping dependency installation by request.")

    print("Environment setup completed successfully.")
    print(f"Use this interpreter to run the app: {venv_python}")


if __name__ == "__main__":
    main()
