from typing import NamedTuple, Sequence

from .specs import RecipeSpec

FS = NamedTuple(
    "FS",
    [
        ("args", dict[str, str]),
        ("command", Sequence[str]),
        ("source_id", str),
        ("rtype", str | None),
    ],
)


def recipe_to_fs(recipe: RecipeSpec) -> FS:
    args = {}
    source_id = recipe.source_id
    command: list[str] = []
    for chunk in recipe.command:
        if "[@p:" in chunk and "]" in chunk:
            i, j = chunk.index("[@"), chunk.index("]")
            arg = chunk[i + 4 : j]
            args[arg] = f"RequiredParameter[{arg}]"
        elif "[@o:" in chunk and "]" in chunk:
            i, j = chunk.index("[@"), chunk.index("]")
            arg = chunk[i + 4 : j]
            args[arg] = f"Observable[{arg}]"
        command.append(chunk.replace(f"[@p:", f"[@").replace(f"[@o:", f"[@"))
    for arg in args.keys():
        for chunk in command:
            chunk = chunk.replace(f"[@p:{arg}]", f"[@{arg}]").replace(f"[@o:{arg}]", f"[@{arg}]")
    return FS(command=command, args=args, source_id=source_id, rtype="list[str]")
