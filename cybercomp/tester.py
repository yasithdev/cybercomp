import ast
import astor


def generate_class(
    class_name: str,
    docstring: str,
    required_params: list[str],
    required_paramtypes: list[str],
    optional_params: dict[str, str | None],
    optional_paramtypes: list[str],
    function_name: str = None,
    function_args: dict[str, str] = None,
    function_command: list[str] = None,
    function_return_type: str = None,
):
    # Ensure all lists have the same length
    if len(optional_params) != len(optional_paramtypes):
        raise ValueError("Attributes with defaults and their types must have the same length.")
    if len(required_params) != len(required_paramtypes):
        raise ValueError("Attributes without defaults and their types must have the same length.")

    # Create the class definition node
    class_def = ast.ClassDef(name=class_name, bases=[], body=[], decorator_list=[])

    # Add docstring if provided
    if docstring:
        class_def.body.append(ast.Expr(value=ast.Constant(value=docstring)))

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

    # Create __init__ method
    init_method = ast.FunctionDef(
        name="__init__",
        args=ast.arguments(
            args=[ast.arg(arg="self", annotation=None)]
            + [
                ast.arg(arg=attr, annotation=ast.Name(id=typ, ctx=ast.Load()))
                for attr, typ in zip(required_params, required_paramtypes)
            ]
            + [
                ast.arg(arg=attr, annotation=ast.Name(id=typ, ctx=ast.Load()))
                for attr, typ in zip(optional_params.keys(), optional_paramtypes)
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[ast.Constant(value=val) for val in optional_params.values()],
        ),
        body=[],
        decorator_list=[],
    )

    # Add assignments in __init__ method
    for attr in required_params:
        init_method.body.append(
            ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Store()), attr=attr, ctx=ast.Store())],
                value=ast.Name(id=attr, ctx=ast.Load()),
            )
        )

    for attr in optional_params.keys():
        init_method.body.append(
            ast.Assign(
                targets=[ast.Attribute(value=ast.Name(id="self", ctx=ast.Store()), attr=attr, ctx=ast.Store())],
                value=ast.Name(id=attr, ctx=ast.Load()),
            )
        )

    # Add __init__ method to class body
    class_def.body.append(init_method)

    

    # Create the AST module and add the class definition
    module = ast.Module(body=[class_def], type_ignores=[])

    # Convert AST to source code
    class_code = astor.to_source(module).strip()  # Use .strip() to clean up any leading/trailing whitespace

    return class_code


# Example usage
class_name = "MyClass"
docstring = "This class demonstrates the addition of a function with specific arguments and commands."
required_params = ["height", "weight"]
required_paramtypes = ["float", "float"]
optional_params = {"name": "John Doe", "age": 30}
optional_paramtypes = ["str", "int"]

function_name = "make_network"
function_args = {"x": "str", "y": "int"}
function_command = ["setx=[@x]", "sety=[@y]", "run", "code"]
function_return_type = "list[str]"

code = generate_class(
    class_name,
    docstring,
    required_params,
    required_paramtypes,
    optional_params,
    optional_paramtypes,
    function_name=function_name,
    function_args=function_args,
    function_command=function_command,
    function_return_type=function_return_type,
)
print(code)
