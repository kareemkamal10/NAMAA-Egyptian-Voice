import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("ceatevenv")


def setup_logger(project_dir: Path, log_name: str = "ceatevenv") -> Path:
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{log_name}.log"
    if log_file.exists():
        log_file.unlink()

    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    return log_file


def run_cmd(command: list[str], cwd: Path | None = None) -> str:
    cmd_str = " ".join(str(part) for part in command)
    LOGGER.info(f"> {cmd_str}")

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: list[str] = []
    if process.stdout is None:
        raise RuntimeError("Unable to capture command output.")

    for line in process.stdout:
        line = line.rstrip("\n")
        captured_lines.append(line)
        if line.strip():
            LOGGER.info(line)

    process.wait()
    output_text = "\n".join(captured_lines)

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed ({process.returncode}): {cmd_str}\n"
            f"{output_text[-2500:]}"
        )
    return output_text


def get_venv_python(venv_path: Path) -> Path:
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def ensure_pip_available(python_executable: str, cwd: Path):
    try:
        run_cmd([python_executable, "-m", "pip", "--version"], cwd=cwd)
        return
    except RuntimeError:
        LOGGER.info("pip is not available. Trying ensurepip...")

    run_cmd([python_executable, "-m", "ensurepip", "--upgrade"], cwd=cwd)


def ensure_virtualenv_available(python_executable: str, cwd: Path):
    try:
        run_cmd([python_executable, "-m", "virtualenv", "--version"], cwd=cwd)
        return
    except RuntimeError:
        LOGGER.info("virtualenv is not available. Installing it...")

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
            LOGGER.info("Built-in venv failed; falling back to virtualenv...")
            LOGGER.info(str(venv_error))

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
        "--log-name",
        default="ceatevenv",
        help="Log file name (without extension).",
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

    log_file = setup_logger(project_dir=project_dir, log_name=args.log_name)
    LOGGER.info("Start setup_env")
    LOGGER.info(f"Logging to: {log_file}")

    venv_path = Path(args.venv_path)
    if not venv_path.is_absolute():
        venv_path = (project_dir / venv_path).resolve()

    requirements_path = Path(args.requirements)
    if not requirements_path.is_absolute():
        requirements_path = (project_dir / requirements_path).resolve()

    python_executable = (args.python or sys.executable).strip()
    if not python_executable:
        raise RuntimeError("No Python interpreter selected for virtualenv creation.")

    LOGGER.info(f"Project directory: {project_dir}")
    LOGGER.info(f"Virtualenv path : {venv_path}")
    LOGGER.info(f"Requirements    : {requirements_path}")
    LOGGER.info(f"Python          : {python_executable}")
    LOGGER.info(f"Venv backend    : {args.venv_backend}")

    if args.recreate and venv_path.exists():
        LOGGER.info(f"Removing existing virtualenv: {venv_path}")
        shutil.rmtree(venv_path)

    if not venv_path.exists():
        create_virtual_environment(
            python_executable=python_executable,
            venv_path=venv_path,
            backend=args.venv_backend,
            cwd=project_dir,
        )
    else:
        LOGGER.info("Virtualenv already exists. Reusing it.")

    venv_python = get_venv_python(venv_path)
    if not venv_python.exists():
        raise FileNotFoundError(f"Virtualenv python not found: {venv_python}")

    run_cmd([str(venv_python), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], cwd=project_dir)

    if not args.skip_install:
        if not requirements_path.exists():
            raise FileNotFoundError(f"Requirements file not found: {requirements_path}")
        run_cmd([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)], cwd=project_dir)
    else:
        LOGGER.info("Skipping dependency installation by request.")

    LOGGER.info("Environment setup completed successfully.")
    LOGGER.info(f"Use this interpreter to run the app: {venv_python}")
    LOGGER.info("End setup_env")


if __name__ == "__main__":
    main()
