import ast
import os


def extract_functions_and_classes(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    functions_and_classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            args = [arg.arg for arg in node.args.args]
            functions_and_classes.append(
                f"def {func_name}({', '.join(args)}):")
        elif isinstance(node, ast.ClassDef):
            class_name = node.name
            bases = [base.id for base in node.bases if isinstance(
                base, ast.Name)]
            class_signature = f"class {class_name}({', '.join(bases)}):" if bases else f"class {class_name}:"
            functions_and_classes.append(class_signature)

    return functions_and_classes


def write_to_txt(functions_and_classes, txt_file_path):
    with open(txt_file_path, 'w') as file:
        for item in functions_and_classes:
            file.write(item + "\n")


def main():
    # Replace with your file path
    file_path = '/home/sanjotst/mPLUG-Owl/mPLUG-Owl2/mplug_owl2/model/modeling_mplug_owl2.py'
    functions_and_classes = extract_functions_and_classes(file_path)

    # Generate output file name based on input file name
    base_name = os.path.basename(file_path)
    name, _ = os.path.splitext(base_name)
    txt_file_path = f"{name}_functions_and_classes.txt"

    write_to_txt(functions_and_classes, txt_file_path)
    print(f"Functions and classes have been written to {txt_file_path}")


if __name__ == "__main__":
    main()
