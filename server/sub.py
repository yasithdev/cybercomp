import argparse
import shutil
from pathlib import Path
from typing import Any

import chevron
import pydantic
import yaml


class Template(pydantic.BaseModel):
    """
    Template Model

    """

    file: str
    params: dict[str, Any]


def substitute_single(
    source_file: Path,
    defaults: dict[str, Any],
    args: dict[str, Any] = {},
) -> str:
    """
    Substitute a single template file with given arguments.

    @param source_file: Path to the template file
    @param defaults: Default arguments to substitute
    @args: Override arguments to substitute

    """
    with open(source_file) as f:
        template = f.read()
    sub_params = defaults.copy()
    sub_params.update(args)
    return chevron.render(template, sub_params)


def substitute_all(
    source_dir: Path,
    target_dir: Path,
    args: dict[str, Any],
) -> None:
    """
    Substitute all templates in a source directory with given arguments.

    @source_id: Id of the source directory
    @args: Arguments to substitute

    """

    # copy source_dir to target_dir
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("template.cc.yml"))
    print(f"[Copy] {source_dir} to {target_dir} - OK")

    # load default args, if present
    with open(source_dir / f"template.cc.yml") as f:
        templates = [Template(**item) for item in yaml.safe_load(f)]

    # overwrite target_dir with generated files
    for template in templates:
        # substitute source template with given params
        content = substitute_single(source_dir / template.file, template.params, args)
        # store generate file in target_dir
        with open(target_dir / template.file, "w") as f:
            f.write(content)
        print(f"[Sub] {target_dir / template.file} - OK")
    print("DONE")


if __name__ == "__main__":

    # Define parser
    parser = argparse.ArgumentParser(description="Generate an executable codebase from a templated codebase")
    parser.add_argument("-s", "--source_id", required=True, help="Id of Source Code")
    parser.add_argument("-a", "--args_file", required=False, help="Path to Args File (YAML)")

    # Get arguments from parser
    args = parser.parse_args()
    source_id = args.source_id
    args_file = Path(args.args_file)
    args = {}
    if args_file.exists() and args_file.is_file() and args_file.suffix == ".yml":
        with open(args_file) as f:
            args = yaml.safe_load(f)
        for k, v in args.items():
            print(f"[Param] {k}: {v}")

    # define base_dir and target_dir based on source_id
    db_dir = Path("database")
    source_dir = db_dir / "sources" / source_id
    target_dir = db_dir / "targets" / source_id

    substitute_all(source_dir, target_dir, args)
