import shutil
from pathlib import Path
from typing import List

import nox

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["tests"]

SRCS = ("aidial_rag", "tests", "noxfile.py")


@nox.session()
def test(session):
    """Runs tests"""
    refresh = "--refresh" in session.posargs
    args = session.posargs
    if refresh:
        print("The cache would be refreshed")
        session.env["REFRESH"] = "true"
        args.remove("--refresh")

    session.run("poetry", "install", "--only", "main, test", external=True)
    session.run(
        "python",
        "-m",
        "nltk.downloader",
        "stopwords",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    )
    session.run("pytest", *args, env=session.env)


@nox.session()
def eval(session):
    """Runs RAG evaluation"""
    session.run("poetry", "install", "--with", "eval", external=True)
    session.run(
        "python",
        "-m",
        "nltk.downloader",
        "stopwords",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    )
    session.run("python", "./eval/eval_retriever.py")


@nox.session()
def update_docs(session):
    session.run("poetry", "install", "--with", "doc", external=True)
    session.run(
        "settings-doc",
        "generate",
        "--class",
        "aidial_rag.app_config.AppConfig",
        "--output-format",
        "markdown",
        "--update",
        "README.md",
        "--between",
        "<!-- generated-app-config-env-start -->",
        "<!-- generated-app-config-env-end -->",
        "--heading-offset",
        "4",
    )


@nox.session()
def lint(session):
    """Runs linters and typecheckers"""
    session.run("poetry", "install", "--with", "lint", external=True)
    args = session.posargs or SRCS
    session.run("ruff", "check", *args)
    session.run("ruff", "format", "--check", *args)
    session.run("pyright", *args)


@nox.session()
def format(session: nox.Session):
    """Runs formatters to fix linting errors"""
    session.run("poetry", "install", "--only", "lint", external=True)
    args = session.posargs or SRCS
    session.run("ruff", "check", "--fix", *args)
    session.run("ruff", "format", *args)


@nox.session(python=False)
def clean(_: nox.Session):
    """Cleans up the project by removing unnecessary files and directories"""

    def remove_dir(directory_path: Path):
        if directory_path.is_dir():
            shutil.rmtree(directory_path)
            print(f"Removed: {directory_path}")

    def remove_recursively(pattern: str, exclude_dirs: List[str] | None = None):
        if exclude_dirs is None:
            exclude_dirs = []

        def is_excluded(file: Path) -> bool:
            return any(dir in str(file) for dir in exclude_dirs)

        for file in Path(".").rglob(pattern):
            if not is_excluded(file):
                remove_dir(file)

    remove_dir(Path(".nox"))
    remove_dir(Path("dist"))
    remove_recursively("__pycache__", exclude_dirs=[".venv"])
    remove_recursively(".pytest_cache", exclude_dirs=[".venv"])
