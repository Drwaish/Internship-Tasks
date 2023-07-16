import glob
import module

def main ():
    ''''
    Driver code of text editor

    Parameters
    ----------
    None

    Return
    ------
    None

    '''
try:
    while True:
        print('''
                    Press o for Open File.
                    Press e for Edit File
                     ''')
        options = input("Select from Menu: ")
        if options == 'o':
            print("Following are the files")
            print(glob.glob('*.*'))
            filename = input("Enter file name: ")
            # The argument to the function may be any descriptive text
            print(module.open_file(filename = filename))
            input("Press the Enter key to continue: ")

        elif options == 'e':
            filename = input("Enter file name or path: ")
            module.edit_file(filename = filename)
        else:
            print("Please Enter valid options")
except EOFError:
    print("Bye")
