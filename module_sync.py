"""File synchronization tool"""
import os
import module



def sync_file(content : str, file_path : list) -> None:
    """
    Sync file that stores on different.

    Parameters
    ----------
    filename
        Name of the file 
    content
        Content add in file
    filepath:
        Where other file are stored
    
    Return
    ------
    None
    """
    for f_path in file_path:
        if os.path.isfile(f_path): #if file exist than open and sync data
            with open(f_path,'w',encoding='Utf-8') as file:
                file.writelines(content)
            file.close()
        #if file not exist delete that address
        else:
            file_path.remove(f_path)

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
    return module.open_file(filename)
def edit_file(filename : str, file_paths : list[str]) -> None:

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
        with open(file = filename, mode = 'r', encoding = 'Utf-8') as file:
            content_str.append(file.readlines())
        file.close()
        content='\n'.join(content_str[0])
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
        #module.save_file(filename,content)
        sync_file(content = content, file_path = file_paths)

def add_new_file(file_name: list, file_path : list[list[str]]) -> None:
    """
    User want more file for maintaining synchronization

    Parameters
    ----------
    file_name
        Contain name of files 
    file_path
        Contain adress of files in corresponding index
    
    Return
    ------
    None
    """
    try:
        name = input("Enter name of file : ")
        file_name.append(name)
        num=int(input("Enter How many file locations you want to enter : "))
        address=[]
        for i in range(num):
            addr = input("Enter the address of file location ")
            address.append(addr)
        file_path.append(address)
        print(file_path)

    except NameError as except_error:
        print(except_error)
    except ValueError as except_error:
        print(except_error)
    except RuntimeError as error:
        print(error)
    else:
        print("ERROR")
