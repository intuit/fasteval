"""Tests for fasteval.utils.skill_parsing."""

from pathlib import Path

import pytest

from fasteval.utils.skill_parsing import flatten_skill_folder


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    skill = tmp_path / "sample-skill"
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("secret", encoding="utf-8")
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text("secret", encoding="utf-8")
    (skill / "references").mkdir(parents=True)
    (skill / "scripts").mkdir()
    (skill / "assets").mkdir()
    (skill / ".claude").mkdir()
    (skill / "examples").mkdir()
    skill_md = """---
name: sample-skill
description: Sample & test
---

Use <carefully>

"""
    (skill / "SKILL.md").write_text(
        skill_md,
        encoding="utf-8",
    )
    (skill / "references" / "guide.md").write_text("Read A & B.\n", encoding="utf-8")
    (skill / "scripts" / "run.py").write_text('print("ok")\n', encoding="utf-8")
    (skill / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff")
    (skill / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    (skill / "notes.md").write_text("Do not include.", encoding="utf-8")
    (skill / "example.txt").write_text("Do not include.", encoding="utf-8")
    (skill / "examples" / "demo.txt").write_text("Example.", encoding="utf-8")
    (skill / "references" / "linked.txt").symlink_to(outside_file)
    (skill / "linked-directory").symlink_to(outside_directory, target_is_directory=True)
    return skill


def test_flattens_skill_directory_as_xml(skill_dir: Path):
    path_result = flatten_skill_folder(skill_dir)
    string_result = flatten_skill_folder(str(skill_dir))

    assert path_result == string_result
    assert path_result == """<skill>
  <file path="SKILL.md">
---
name: sample-skill
description: Sample &amp; test
---

Use &lt;carefully&gt;


  </file>
  <resource path="assets/logo.png" />
  <file path="references/guide.md">
Read A &amp; B.

  </file>
  <file path="scripts/run.py">
print("ok")

  </file>
</skill>"""
    assert "linked.txt" not in path_result
    assert "secret" not in path_result


def test_includes_additional_resources(skill_dir: Path):
    string_result = flatten_skill_folder(
        skill_dir,
        additional_resource_paths=[
            "notes.md",
            "examples",
            "references",
            "references/guide.md",
        ],
    )
    path_result = flatten_skill_folder(
        skill_dir,
        additional_resource_paths=[
            Path("notes.md"),
            Path("examples"),
            Path("references"),
            Path("references/guide.md"),
        ],
    )

    assert string_result == path_result
    for result in (string_result, path_result):
        assert '<file path="examples/demo.txt">' in result
        assert '<file path="notes.md">' in result
        assert result.count('<file path="references/guide.md">') == 1


@pytest.mark.parametrize(
    ("additional_path", "message"),
    [
        ("/tmp/outside.txt", "Additional resource path must be relative"),
        ("../outside.txt", "Additional resource path must be relative"),
        ("missing.txt", "Additional resource path does not exist"),
        (
            "references/linked.txt",
            "Additional resource path must not be a symbolic link",
        ),
        (
            "linked-directory/secret.txt",
            "Additional resource path must resolve within the skill directory",
        ),
    ],
)
def test_rejects_invalid_additional_resource_paths(
    skill_dir: Path,
    additional_path: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        flatten_skill_folder(
            skill_dir,
            additional_resource_paths=[additional_path],
        )


def test_rejects_path_that_is_not_a_directory(tmp_path: Path):
    missing_path = tmp_path / "missing"

    with pytest.raises(
        ValueError, match=f"Skill path must be a directory: {missing_path}"
    ):
        flatten_skill_folder(missing_path)


def test_rejects_directory_without_root_skill_file(tmp_path: Path):
    with pytest.raises(
        ValueError, match=f"Skill directory must contain SKILL.md: {tmp_path}"
    ):
        flatten_skill_folder(tmp_path)


def test_rejects_symlinked_root_skill_file(skill_dir: Path):
    outside = (skill_dir / "references" / "linked.txt").resolve()
    (skill_dir / "SKILL.md").unlink()
    (skill_dir / "SKILL.md").symlink_to(outside)

    with pytest.raises(
        ValueError, match=f"Skill directory must contain SKILL.md: {skill_dir}"
    ):
        flatten_skill_folder(skill_dir)


def test_skips_symlinked_default_resource_directory(skill_dir: Path):
    outside = (skill_dir / "linked-directory").resolve()
    assets = skill_dir / "assets"
    (assets / "logo.png").unlink()
    assets.rmdir()
    assets.symlink_to(outside, target_is_directory=True)

    result = flatten_skill_folder(skill_dir)

    assert "secret" not in result
    assert "assets/secret.txt" not in result
