import re
import pickle

# Load the predefined function signatures from the pickle file
with open('./data/identifier2python.pkl', 'rb') as fp:
    identifier2python = pickle.load(fp)

import re


def fix_function_signature(func_code: str) -> str:
    func_code = re.sub(r'(\s*)def\s', 'def ', func_code)
    func_code = re.sub(r'(\s*)"""', r'\1"""', func_code)

    func_code = re.sub(r'(\s*)"""(.+?)"""', lambda match: match.group(0).replace('\n', '\n' + match.group(1)),
                       func_code, flags=re.DOTALL)

    func_code = re.sub(r'(\s*)pass', r'\1pass', func_code)

    return func_code


def extract_function_calls(code: str):
    """
    Extracts all function calls and their arguments from the provided code using regex.

    Args:
    - code (str): The input Python code to extract function calls from.

    Returns:
    - list: A list of tuples, each containing the function name and its argument string.
    """
    pattern = r'(\w+)\s*\((.*?)\)'  # Regex to match function calls and arguments
    calls = re.findall(pattern, code)
    return calls


def check_workflow_code_validity(code: str) -> str:
    """
    Validates a workflow code by checking the consistency of function calls and parameters.
    - Extracts function calls from the code.
    - Checks if the parameters match the predefined signatures by executing the code.

    Args:
    - code (str): The workflow code to check.

    Returns:
    - str: A message indicating whether the code is valid or describing the error.
    """
    if len(code.strip()) == 0:
        return "The code is empty."


    # Step 1: Extract function calls from the code
    function_calls = extract_function_calls(code)

    if not function_calls:
        return "No function calls found. The code might be empty or improperly formatted."

    # Step 2: Check each function call against its corresponding signature
    for function_name, args in function_calls:
        # Step 3: Check if the function name exists in the predefined dictionary
        expected_code = identifier2python.get(function_name)

        if expected_code is None:
            return f"Function {function_name} not found in the predefined function signatures."

        # Step 4: Build the full function definition by concatenating the signature with the actual function call
        full_code = fix_function_signature(expected_code.strip()) + f"\n{function_name}({args})"

        # Step 5: Try to execute the combined code
        try:
            # Use exec to compile and execute the code dynamically
            exec(full_code)

        except Exception as e:
            # Catch any error that happens during execution and return the error message
            return f"Error processing function {function_name}: {str(e)}"

    # If all function calls are valid, return a success message
    return "Workflow code is valid, all function calls match the expected parameter formats."



if __name__ == "__main__":
    # Sample Workflow Code for Testing
    workflow_code_1 = """
    is_workflow_actions_getlatestlivephotos(3)
    """

    workflow_code_2 = """
    is_workflow_actions_openurl("https://example.com")
    """

    workflow_code_3 = """
    is_workflow_actions_getlatestlivephotos()
    is_workflow_actions_openurl(code = "https://example.com")
    """

    workflow_code_4 = """
    is_workflow_actions_getlatestlivephotos("incorrect_type")
    """

    workflow_code_5 = """
    is_workflow_actions_getlatestlivephotos(3)
    is_workflow_actions_openurl(12345, 567)
    """

    # Run the validation on different workflow code examples
    result_1 = check_workflow_code_validity(workflow_code_1)
    result_2 = check_workflow_code_validity(workflow_code_2)
    result_3 = check_workflow_code_validity(workflow_code_3)
    result_4 = check_workflow_code_validity(workflow_code_4)
    result_5 = check_workflow_code_validity(workflow_code_5)

    # Print results for each test case
    print("Test Case 1 Result:", result_1)
    print("Test Case 2 Result:", result_2)
    print("Test Case 3 Result:", result_3)
    print("Test Case 4 Result:", result_4)
    print("Test Case 5 Result:", result_5)
