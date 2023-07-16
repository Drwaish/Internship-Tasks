import glob
import module_sync


file_name=['temp.txt','def.txt']
file_path=[["D:/Intern Training/July_14_Task/Task3/def.txt",
            "D:/Intern Training/July_14_Task/Task2/def.txt"],
           ["d:/Intern Training/July_14_Task/Task1/temp.txt",
            "d:/Intern Training/July_14_Task/Task4/temp.txt"]]
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
                    Press e for Edit File.
                    Press f for add new path
                     ''')
        options = input("Select from Menu: ")
        if options == 'o':
            print("Following are the files")
            print(glob.glob('*.*'))
            filename = input("Enter file name: ")
            # The argument to the function may be any descriptive text
            print(module_sync.open_file(filename = filename))
            input("Press the Enter key to continue: ")

        elif options == 'e':
            filename = input("Enter file name or path: ")
            index_number= file_name.index(filename)
            module_sync.edit_file(filename = filename, file_paths=file_path[index_number])
        elif options == 'f':
            module_sync.add_new_file(file_name=file_name,file_path=file_path)
        else:
            print("Please Enter valid options")
except EOFError:
    print("Bye")

if __name__ =="__main__":
    main()
