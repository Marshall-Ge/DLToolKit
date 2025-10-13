import os
import ast
from collections import defaultdict

class RegisterModels:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.model_dir = os.path.join(project_dir, 'dltoolkit/models')
        self.utils_init_path = os.path.join(project_dir, 'dltoolkit/utils', '__init__.py')
        os.makedirs(os.path.dirname(self.utils_init_path), exist_ok=True)
        self.registered_classes = defaultdict(list)  # {file_path: [class_name1, class_name2, ...]}

    def __call__(self):
        assert os.path.isdir(self.model_dir), f"Model directory not found: {self.model_dir}"
        import_statements = []
        model_dir_name = os.path.basename(self.model_dir)

        self._collect_registered_classes()

        for file_path, classes in self.registered_classes.items():
            rel_path = os.path.relpath(file_path, self.model_dir)
            module_parts = os.path.splitext(rel_path)[0].split(os.sep)
            module_path = ".".join(module_parts)

            for cls in classes:
                import_stmt = f"from dltoolkit.{model_dir_name}.{module_path} import {cls}"
                import_statements.append(import_stmt)

        import_statements = sorted(list(set(import_statements)))

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

        print(f"Updated {self.utils_init_path}, been imported {len(import_statements)} classes.")

    def _collect_registered_classes(self):
        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    classes = self._parse_registered_classes(file_path)
                    if classes:
                        self.registered_classes[file_path] = classes

    def _parse_registered_classes(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            registered_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if self._has_model_registry_decorator(node):
                        registered_classes.append(node.name)
            return registered_classes
        except Exception as e:
            print(f"Parse {file_path} failed: {e}")
            return []

    def _has_model_registry_decorator(self, class_node):
        for decorator in class_node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    if (isinstance(decorator.func.value, ast.Name) and
                        decorator.func.value.id == 'MODEL_REGISTRY' and
                        decorator.func.attr == 'register'):
                        return True
        return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    registrar = RegisterModels(project_dir)
    registrar()

if __name__ == "__main__":
    main()