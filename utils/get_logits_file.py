import os
def find_files_with_string(directory, search_string):
   # Traverse through the directory
   for root, dirs, files in os.walk(directory):
       for file in files:
           if file.endswith(".py"):
               file_path = os.path.join(root, file)
               with open(file_path, 'r', encoding='utf-8') as f:
                   contents = f.read()
                   if search_string in contents:
                       print(f"'{search_string}' found in: {file_path}")
# Specify the directory and the string to search for
subdirectory = '/path/to/your/subdirectory'  # replace with your directory path
search_string = 'logits'
# Call the function
find_files_with_string(subdirectory, search_string)