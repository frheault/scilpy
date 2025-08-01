import os
import ast
import re

def get_functions_from_py_file(filepath):
    """
    Parses a Python file and returns a list of function and method names.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    functions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Ignore private methods
                if not node.name.startswith('_'):
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef):
                        # Ignore private methods
                        if not sub_node.name.startswith('_'):
                            functions.append(f"{node.name}.{sub_node.name}")
    except SyntaxError as e:
        print(f"Could not parse {filepath}: {e}")
    return functions

def get_functions_from_pyx_file(filepath):
    """
    Parses a Cython file and returns a list of function names using regex.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    functions = re.findall(r'^(?:c|cp)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)', content, re.MULTILINE)
    return [f for f in functions if not f.startswith('_')]

def get_test_functions(filepath):
    """
    Parses a Python test file and returns a list of test function names.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    functions = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                functions.append(node.name)
    except SyntaxError as e:
        print(f"Could not parse {filepath}: {e}")
    return functions

def find_test_files(module_path):
    """
    Finds the test file(s) for a given module.
    Returns a list of paths.
    """
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path).split('.')[0]
    test_dir = os.path.join(module_dir, 'tests')

    if not os.path.exists(test_dir):
        return []

    test_files = []
    for f in os.listdir(test_dir):
        if f.startswith('test_') and f.endswith('.py'):
            # Heuristic: if test file name contains module name (without .py)
            if module_name in f.replace('.py', ''):
                test_files.append(os.path.join(test_dir, f))

    return test_files

def main():
    for root, dirs, files in os.walk('scilpy'):
        if 'tests' in dirs:
            # This will not work as intended if tests are not direct subdirs
            pass

        for file in files:
            if root.endswith('/tests'):
                continue

            if file.endswith(('.py', '.pyx')) and not file.startswith('__'):
                module_path = os.path.join(root, file)

                if module_path == 'scilpy/setup.py':
                    continue

                if module_path.endswith('.py'):
                    module_functions = get_functions_from_py_file(module_path)
                else:
                    module_functions = get_functions_from_pyx_file(module_path)

                if not module_functions:
                    continue

                test_files = find_test_files(module_path)

                if not test_files:
                    print(f"\n--- Module: {module_path} ---")
                    print("  No test file found.")
                    for func in module_functions:
                        print(f"  - Missing test for: {func}")
                    continue

                all_test_functions = []
                for test_file in test_files:
                    all_test_functions.extend(get_test_functions(test_file))

                missing_tests = []
                for func in module_functions:
                    # Very simple mapping. A better way would be needed for robust check.
                    # Ex: func -> test_func or func -> test_module_func
                    found = any(func.lower() in test.lower() for test in all_test_functions)
                    if not found:
                        missing_tests.append(func)

                if missing_tests:
                    print(f"\n--- Module: {module_path} ---")
                    print(f"  Test files: {test_files}")
                    for func in missing_tests:
                        print(f"  - Missing test for: {func}")

if __name__ == '__main__':
    main()
