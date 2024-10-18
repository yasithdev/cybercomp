import argparse
import shutil
from pathlib import Path
from typing import Any

import chevron
import pydantic
import yaml


class Substitution(pydantic.BaseModel):
    file: str
    params: dict[str, Any]


def run_substitution(
    file: Path,
    default_params: dict[str, Any],
    override_params: dict[str, Any] = {},
) -> str:
    """
    Substitute template file with given parameters.

    @param file: Path to the template file
    @param default_params: Default parameters to substitute
    @override_params: Override parameters to substitute

    """
    with open(file) as f:
        template = f.read()
    sub_params = default_params.copy()
    sub_params.update(override_params)
    return chevron.render(template, sub_params)


if __name__ == "__main__":

    # Define parser
    parser = argparse.ArgumentParser(description="Generate source code from a parametrized code template")
    parser.add_argument("-s", "--source_id", required=True, help="Id of Source Code")
    parser.add_argument("-p", "--params_file", required=False, help="Path to Override Parameter File (YAML)")

    # Get arguments from parser
    args = parser.parse_args()
    source_id = args.source_id
    params_file = Path(args.params_file)

    override_params = {}
    if params_file.exists() and params_file.is_file() and params_file.suffix == ".yml":
        with open(params_file) as f:
            override_params = yaml.safe_load(f)
        for k, v in override_params.items():
            print(f"[Param] {k}: {v}")

    # define base_dir and target_dir based on source_id
    db_dir = Path("database")
    source_dir = db_dir / "sources" / source_id
    target_dir = db_dir / "targets" / source_id

    # copy source_dir to target_dir
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("template.cc.yml"))
    print(f"[Task] Copy {source_dir} to {target_dir} - OK")

    # overwrite target_dir with generated files
    with open(source_dir / f"template.cc.yml") as f:
        data = yaml.safe_load(f)
        for sub in data:
            sub = Substitution(**sub)
            template_file = source_dir / sub.file
            # substitute source template with given params
            content = run_substitution(template_file, sub.params, override_params)
            # store generate file in target_dir
            with open(target_dir / sub.file, "w") as f:
                f.write(content)
            print(f"[Task] Generate {target_dir / sub.file} - OK")
    print("DONE")
