import os
import ast

class RegisterModels:
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.model_dir = os.path.join(script_dir, 'dltoolkit/models')
        self.utils_init_path = os.path.join(script_dir, 'dltoolkit/utils', '__init__.py')
        os.makedirs(os.path.dirname(self.utils_init_path), exist_ok=True)

    def __call__(self):
        assert os.path.isdir(self.model_dir)
        import_statements = []
        model_dir_name = os.path.basename(self.model_dir)

        for root, dirs, files in os.walk(self.model_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)

                    rel_path = os.path.relpath(file_path, self.model_dir)
                    module_parts = os.path.splitext(rel_path)[0].split(os.sep)
                    module_path = ".".join(module_parts)

                    classes = self._find_torch_nn_classes(file_path)

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
            # f.write(''.join(existing_content))

            f.write(header)
            for stmt in import_statements:
                f.write(f"{stmt}\n")
            f.write(footer)

        print(f"Updated {self.utils_init_path}，has import {len(import_statements)} number of classes")

    def _find_torch_nn_classes(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())

            nn_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
                            if base.value.id == 'nn' and hasattr(base, 'attr'):
                                nn_classes.append(node.name)
                        elif isinstance(base, ast.Name) and base.id in ['Module', 'Sequential', 'ModuleList']:
                            nn_classes.append(node.name)

            return list(set(nn_classes))
        except Exception as e:
            print(f"{e}")
            return []


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    registrar = RegisterModels(script_dir)
    registrar()

if __name__ == "__main__":
    main()
