"""Utilities for reading Agent Skills directories."""

from pathlib import Path
from typing import List
from xml.sax.saxutils import escape, quoteattr

DEFAULT_RESOURCE_DIRECTORIES = ("references", "scripts", "assets")


def flatten_skill_folder(
    skill_path: str | Path,
    additional_resource_paths: List[str] | List[Path] | None = None,
) -> str:
    """Return an XML representation of an Agent Skills directory."""
    root = Path(skill_path)
    if not root.is_dir():
        raise ValueError(f"Skill path must be a directory: {root}")
    skill_md = root / "SKILL.md"
    if not skill_md.is_file() or skill_md.is_symlink():
        raise ValueError(f"Skill directory must contain SKILL.md: {root}")

    resource_paths = [skill_md]
    resource_paths.extend(
        path
        for directory in DEFAULT_RESOURCE_DIRECTORIES
        if (path := root / directory).is_dir() and not path.is_symlink()
    )

    for additional_path in additional_resource_paths or []:
        relative_path = Path(additional_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Additional resource path must be relative to the skill directory: "
                f"{additional_path}"
            )

        resource_path = root / relative_path
        if not resource_path.exists():
            raise ValueError(
                f"Additional resource path does not exist: {additional_path}"
            )
        if resource_path.is_symlink():
            raise ValueError(
                f"Additional resource path must not be a symbolic link: "
                f"{additional_path}"
            )
        if not resource_path.resolve().is_relative_to(root.resolve()):
            raise ValueError(
                f"Additional resource path must resolve within the skill directory: "
                f"{additional_path}"
            )
        resource_paths.append(resource_path)

    files = set()
    for resource_path in resource_paths:
        if resource_path.is_file():
            files.add(resource_path)
        elif resource_path.is_dir():
            files.update(
                path
                for path in resource_path.rglob("*")
                if path.is_file() and not path.is_symlink()
            )

    sorted_files = sorted(
        files,
        key=lambda path: (
            path.relative_to(root).as_posix() != "SKILL.md",
            path.relative_to(root).as_posix(),
        ),
    )

    elements = []
    for path in sorted_files:
        relative_posix = path.relative_to(root).as_posix()
        path_attribute = quoteattr(relative_posix)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            elements.append(f"  <resource path={path_attribute} />")
            continue

        elements.append(f"  <file path={path_attribute}>\n{escape(content)}\n  </file>")

    return "<skill>\n" + "\n".join(elements) + "\n</skill>"
