""" Text editor with basic functionalities"""
import os


def create_file() -> None:
    """
    Create file use noname.txt as a filename.

    Parameters
    ----------
    None

    Return
    ------
    None

    """
    with open("noname.txt", 'w', encoding="Utf-8") as file:
        file.close()


def save_file(filename: str, content: str) -> None:
    """
    User will enter file name if file name no enter than use 
    noname.txt as a filename.

    Parameters
    ----------
    filename
        contain name of the file
    content
        contain content of file
    Return
    ------
    None

    """
    # os.rename('noname.txt', filename)
    try:
        print('''
                Press s to Save:
                Press sa to Save as:
                ''')
        options = input("Enter Options here: ")
        if options == 's':
            with open(file=filename, mode='w', encoding="Utf-8") as file:
                file.writelines(content)
            file.close()
        if options == 'sa':
            file_nam = input("Enter file name here: ")
            os.rename(filename, file_nam)
    except FileNotFoundError:
        print('File not found error')


def edit_file(filename: str) -> None:
    """
    User will enter file name to edit file.

    Parameters
    ----------
    filename
        contain name of the file

    Return
    ------
    None

    """
    content = " "
    content_str = []
    try:
        with open(file=filename, mode='r', encoding='Utf-8') as file:
            content_str.append(file.readlines())
        file.close()
        content = '\n'.join(content_str[0])
    except FileNotFoundError:
        print("File not found")
    try:
        while True:
            print('''
                    Press r to replace words:
                    Press a to add conrent:
                 ''')
            print("Ctrl+Z to End this Editing")
            print(content)

            options = input('Enter your Option: ')
            if options == 'r':
                word_to_replace = input(" Enter the word you want to add: ")
                replace_with = input(
                    " Enter the word that you want to change: ")
                content.replace(replace_with, word_to_replace)
            elif options == 'a':
                content_to_append = input(
                    "Enter your content here. For new line enter '\\n' ")
                content += content_to_append
            else:
                print("Kindly Enter Valid Option.Thanks")

    except EOFError:
        save_file(filename, content)


def open_file(filename: str) -> str:
    """
    User will enter file name to edit file.

    Parameters
    ----------
    filename
        contain name of the file

    Return
    ------
    None

    """
    content = ""
    try:
        with open(file=filename, mode='r', encoding='Utf-8') as file:
            content += str(file.readlines())
    except FileNotFoundError:
        print("File not found error")
    return content
