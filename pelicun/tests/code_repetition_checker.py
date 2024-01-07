"""
A utility script for detecting duplicated blocks of lines across
Python test files.
"""

from glob2 import glob


def main(file):
    """
    Identifies and displays repeated consecutive line blocks within a
    file, including their line numbers.

    Args:
    file: Path to the file to be checked for duplicates.
    """
    # file = 'tests/test_uq.py'
    group = 15  # find repeated blocks this many lines

    with open(file, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    num_lines = len(contents)
    for i in range(0, num_lines, group):
        glines = contents[i : i + group]
        for j in range(i + 1, num_lines):
            jlines = contents[j : j + group]
            if glines == jlines:
                print(f'{i, j}: ')
                for k in range(group):
                    print(f'    {jlines[k]}', end='')
                print()


def all_test_files():
    """
    Searches for all Python test files in the 'tests' directory and
    runs the main function to find and print repeated line blocks in each file.
    """
    test_files = glob('tests/*.py')
    for file in test_files:
        print()
        print(file)
        print()
        main(file)


if __name__ == '__main__':
    all_test_files()
