import re
from pprint import pprint


def infer_substituted_params(template, generated_output):
    # Step 1: Identify the Mustache placeholders ({{variable}}) in the template
    placeholders = re.findall(r"{{\s*([\w\.]+)\s*}}", template)

    # Step 2: Replace placeholders in the template with a regex pattern to capture values in the generated output
    regex_pattern = re.escape(template)
    for placeholder in placeholders:
        regex_pattern = regex_pattern.replace(re.escape("{{" + placeholder + "}}"), r"(.+?)")

    # Step 3: Match the generated output with the regex pattern
    match = re.match(regex_pattern, generated_output)
    if match:
        values = match.groups()
        # Step 4: Create a dictionary of placeholders and extracted values
        mapping = {placeholders[i]: values[i] for i in range(len(placeholders))}
        return mapping
    else:
        return None  # No match found


# Example usage
template_fp = "database/sources/sleep_stage_transition/in/params_sub.txt"
generated_output_fp = "database/sources/sleep_stage_transition/in/params.txt"

with open(template_fp, "r") as f:
    template = f.read()

with open(generated_output_fp, "r") as f:
    generated_output = f.read()

mapping = infer_substituted_params(template, generated_output)
pprint(mapping)
