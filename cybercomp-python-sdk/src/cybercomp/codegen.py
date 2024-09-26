import ast

import astor

from .util_recipes import FS


def generate_class_py(
    imports: list[tuple[str, str]],
    class_name: str,
    class_bases: list[str],
    docstring: str,
    fixed_params: dict[str, tuple[str, str]] = {},
    fixed_typeless_params: dict[str, str] = {},
    typed_params: dict[str, str] = {},
    required_params: list[str] = [],
    required_paramtypes: list[str] = [],
    optional_params: dict[str, str | None] = {},
    optional_paramtypes: list[str] = [],
    functions: dict[str, FS] = {},
):
    # Ensure all lists have the same length
    if len(optional_params) != len(optional_paramtypes):
        raise ValueError("Attributes with defaults and their types must have the same length.")
    if len(required_params) != len(required_paramtypes):
        raise ValueError("Attributes without defaults and their types must have the same length.")

    import_statements = []
    for mod, imp in imports:
        import_statements.append(ast.ImportFrom(module=mod, names=[ast.alias(name=imp, asname=None)], level=0))

    # Create the class definition node
    class_def = ast.ClassDef(
        name=class_name,
        bases=[ast.Name(id=b, ctx=ast.Load()) for b in class_bases],
        body=[],
        decorator_list=[],
    )  # type: ignore

    # Add docstring if provided
    if docstring:
        class_def.body.append(ast.Expr(value=ast.Constant(value=docstring)))

    # Add fixed-untyped attributes first
    for attr, (typ, default) in fixed_params.items():
        class_def.body.append(
            ast.AnnAssign(
                target=ast.Name(id=attr, ctx=ast.Store()),
                annotation=ast.Name(id=typ, ctx=ast.Load()),
                value=None if default is None else ast.Constant(default),
                simple=1,
            )
        )

    # Add fixed-typed attributes next
    for attr, value in fixed_typeless_params.items():
        class_def.body.append(
            ast.Assign(
                targets=[ast.Name(id=attr, ctx=ast.Store())],
                value=ast.Name(value),
            )
        )

    # Add fixed attributes with defaults first
    for attr, typ in typed_params.items():
        class_def.body.append(
            ast.AnnAssign(
                target=ast.Name(id=attr, ctx=ast.Store()),
                annotation=ast.Name(id=typ, ctx=ast.Load()),
                value=None,
                simple=1,
            )
        )

    # Add class-level attributes without defaults first
    for attr, typ in zip(required_params, required_paramtypes):
        class_def.body.append(
            ast.AnnAssign(
                target=ast.Name(id=attr, ctx=ast.Store()),
                annotation=ast.Name(id=typ, ctx=ast.Load()),
                value=None,
                simple=1,
            )
        )

    # Add class-level attributes with defaults
    for attr, typ in zip(optional_params.keys(), optional_paramtypes):
        class_def.body.append(
            ast.AnnAssign(
                target=ast.Name(id=attr, ctx=ast.Store()),
                annotation=ast.Name(id=typ, ctx=ast.Load()),
                value=None,
                simple=1,
            )
        )

    # Create the function if specified
    for function_name, func in functions.items():
        function_args = func.args
        function_command = func.command
        function_rtype = func.rtype
        function_args_ast = [
            ast.arg(arg=arg, annotation=ast.Name(id=typ, ctx=ast.Load())) for arg, typ in function_args.items()
        ]

        # Add return statement
        if function_rtype:
            function_rtype_ast = ast.Name(id=function_rtype, ctx=ast.Load())
        else:
            function_rtype_ast = ast.Name(id="None", ctx=ast.Load())

        function_elts = []
        for command in function_command:
            if "[@" in command and "]" in command:
                for arg in function_args:
                    command = command.replace(f"[@{arg}]", f"[{arg}]")
                i, j = command.index("["), command.index("]")
                function_elts.append(
                    ast.JoinedStr(
                        [
                            ast.Constant(value=command[:i]),
                            ast.FormattedValue(ast.Name(id=command[i + 1 : j], ctx=ast.Load()), conversion=-1),
                            ast.Constant(value=command[j + 1 :]),
                        ]
                    )
                )
            else:
                function_elts.append(ast.Constant(value=command))

        function_def = ast.FunctionDef(
            name=function_name,
            args=ast.arguments(
                args=[ast.arg(arg="self", annotation=None)] + function_args_ast,
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[],
            ),  # type: ignore
            body=[ast.Return(ast.List(elts=function_elts, ctx=ast.Load()))],
            decorator_list=[],
            returns=function_rtype_ast,
        )

        # Add function to class body
        class_def.body.append(function_def)

    # Create the AST module and add the class definition
    module = ast.Module(body=[*import_statements, class_def], type_ignores=[])

    # Convert AST to source code
    class_code = astor.to_source(module).strip()  # Use .strip() to clean up any leading/trailing whitespace

    return class_code


def generate_module(
    imports: list[tuple[str, str]],
    typedefs: dict[str, str] = {},
):
    # Create import statements for each class
    import_statements = []
    for src, imp in imports:
        import_statements.append(ast.ImportFrom(module=src, names=[ast.alias(name=imp, asname=None)], level=0))

    # Create the AST module with the import statements
    module = ast.Module(body=import_statements, type_ignores=[])

    # add type definitions into
    for name, typ in typedefs.items():
        module.body.append(ast.Assign(targets=[ast.Name(id=name)], value=ast.Name(typ)))

    # Convert AST to source code
    init_code = astor.to_source(module).strip()  # Use .strip() to clean up any leading/trailing whitespace

    return init_code
