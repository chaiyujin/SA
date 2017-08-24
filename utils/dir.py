import os

# find files
def find_files(path, target_ext):
    if target_ext[0] != '.':
        target_ext = '.' + target_ext
    result_list = []
    for parent, dirs, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(os.path.join(parent, file))
            if ext == target_ext:
                result_list.append(name + ext)
    return result_list
