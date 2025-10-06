import os
import ast
from collections import defaultdict

class RegisterModels:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.model_dir = os.path.join(project_dir, 'dltoolkit/models')
        self.utils_init_path = os.path.join(project_dir, 'dltoolkit/utils', '__init__.py')
        os.makedirs(os.path.dirname(self.utils_init_path), exist_ok=True)
        # Cache to store classes and their base classes for each file
        self.class_bases_cache = defaultdict(dict)  # {file_path: {class_name: [base_classes]}}

    def __call__(self):
        assert os.path.isdir(self.model_dir), f"Model directory does not exist: {self.model_dir}"
        import_statements = []
        model_dir_name = os.path.basename(self.model_dir)

        # First collect class information from all files to build cache
        self._collect_all_class_bases()

        # Then analyze each class to check if it inherits from nn.Module (including indirectly)
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    if file_path not in self.class_bases_cache:
                        continue  # Skip files that failed to parse

                    rel_path = os.path.relpath(file_path, self.model_dir)
                    module_parts = os.path.splitext(rel_path)[0].split(os.sep)
                    module_path = ".".join(module_parts)

                    # Get all classes in current file that inherit from nn.Module (including indirectly)
                    nn_module_classes = self._find_nn_module_classes(file_path)

                    for cls in nn_module_classes:
                        import_stmt = f"from dltoolkit.{model_dir_name}.{module_path} import {cls}"
                        import_statements.append(import_stmt)

        # Remove duplicates and sort
        import_statements = sorted(list(set(import_statements)))

        # Generate __init__.py content
        header = "# Automatically generated import statements - Do not modify manually\n"
        footer = "# The automatically generated import statement has ended\n"

        existing_content = []
        if os.path.exists(self.utils_init_path):
            with open(self.utils_init_path, 'r', encoding='utf-8') as f:
                content = f.read()

                if header in content:
                    parts = content.split(header)
                    existing_content.append(parts[0])

                    if footer in parts[1]:
                        parts2 = parts[1].split(footer)
                        existing_content.append(parts2[1])
                else:
                    existing_content.append(content)

        with open(self.utils_init_path, 'w', encoding='utf-8') as f:
            f.write(''.join(existing_content))
            f.write(header)
            for stmt in import_statements:
                f.write(f"{stmt}\n")
            f.write(footer)

        print(f"Updated {self.utils_init_path}, imported {len(import_statements)} classes")

    def _collect_all_class_bases(self):
        """Collect all classes and their base classes from all Python files into cache"""
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    # Parse classes and their base classes in the file
                    class_bases = self._parse_class_bases(file_path)
                    if class_bases:
                        self.class_bases_cache[file_path] = class_bases

    def _parse_class_bases(self, file_path):
        """Parse a single file and return {class_name: [list_of_base_class_names]}"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            class_bases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = []
                    for base in node.bases:
                        # Parse base class name (supports nn.Module or custom class names)
                        base_name = self._get_base_class_name(base)
                        if base_name:
                            bases.append(base_name)
                    class_bases[node.name] = bases
            return class_bases
        except Exception as e:
            print(f"Failed to parse file {file_path}: {e}")
            return {}

    def _get_base_class_name(self, base_node):
        """Extract base class name from AST node (e.g. returns 'Module' for nn.Module, custom class name for user-defined classes)"""
        if isinstance(base_node, ast.Attribute) and isinstance(base_node.value, ast.Name):
            if base_node.value.id == 'nn':
                return base_node.attr  # e.g. nn.Module -> 'Module'
        elif isinstance(base_node, ast.Name):
            return base_node.id  # e.g. custom class MyBaseClass -> 'MyBaseClass'
        return None  # Unsupported base class format (e.g. generics, classes from other modules)

    def _find_nn_module_classes(self, file_path):
        """Find all classes in the file that directly or indirectly inherit from nn.Module"""
        class_bases = self.class_bases_cache.get(file_path, {})
        nn_module_classes = []
        checked = set()  # Avoid infinite recursion from circular inheritance

        def is_subclass_of_nn_module(class_name, current_file):
            """Recursively check if a class inherits from nn.Module"""
            if class_name in checked:
                return False
            checked.add(class_name)

            # Check classes in current file
            if class_name in self.class_bases_cache.get(current_file, {}):
                bases = self.class_bases_cache[current_file][class_name]
                for base in bases:
                    if base == 'Module':  # Directly inherits from nn.Module
                        return True
                    # Recursively check if base class inherits from nn.Module (in current or other files)
                    if self._check_base_in_any_file(base, current_file):
                        return True
            return False

        for class_name in class_bases:
            checked.clear()  # Reset check records
            if is_subclass_of_nn_module(class_name, file_path):
                nn_module_classes.append(class_name)

        return list(set(nn_module_classes))

    def _check_base_in_any_file(self, base_name, current_file):
        """Check if base class is defined in any file and inherits from nn.Module"""
        # First check current file
        if base_name in self.class_bases_cache.get(current_file, {}):
            if self._is_base_module_subclass(base_name, current_file):
                return True
        # Check other files
        for file_path in self.class_bases_cache:
            if file_path == current_file:
                continue
            if base_name in self.class_bases_cache[file_path]:
                if self._is_base_module_subclass(base_name, file_path):
                    return True
        return False

    def _is_base_module_subclass(self, base_name, file_path):
        """Check if the base class in specified file inherits from nn.Module"""
        checked = set()

        def recursive_check(cls_name, cls_file):
            if cls_name in checked:
                return False
            checked.add(cls_name)
            bases = self.class_bases_cache.get(cls_file, {}).get(cls_name, [])
            for b in bases:
                if b == 'Module':  # Directly inherits from nn.Module
                    return True
                # Recursively check base classes of the base class
                if self._check_base_in_any_file(b, cls_file):
                    return True
            return False

        return recursive_check(base_name, file_path)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    registrar = RegisterModels(project_dir)
    registrar()

if __name__ == "__main__":
    main()