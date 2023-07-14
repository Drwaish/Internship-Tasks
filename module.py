'''

Encryption and Decryption of data and aslo file handling.

'''
import random as rd

def generate_random_number() -> int:
    '''
    Random number generator to shift the characters.

    Parameters
    ----------
    None

    Return
    ------
    Integer generate randomly .
    '''
    return rd.randint(0, 26)

def read_file(filename : str) -> list[list[str]]:
    ''''
    Read content of the file.

    Parameters
    ----------
    filename
        Name of the file that needs to be encrypted.
    
    Return
    ------
    List of string in which lines of file populated.

    '''
    file_content=[]
    with open(filename, 'r', encoding = 'Utf-8') as file:
        file_content.append(file.readlines())
    return file_content



def encrypt(text1 : list[list[str]], shift : int) -> list[str]:
    ''''
    Encrypt data with specific shift .
    Parameters
    ----------
    text
        List of strings in which file lines are populated.
    shift
        Shift value of character
    
    Return
    ------
    List of string in which encrypted lines of file populated.
    '''
    encrypted_lines = []
    text = text1[0]
    for lines in text:
        result = ""
        for char in lines:
            if char.isupper():
                result += chr((ord(char) + shift-65) % 26 + 65)
            elif char.islower():
                result += chr((ord(char) + shift-97) % 26 + 97)
            else:
                result+=char
        encrypted_lines.append(result)
    return encrypted_lines

def decrypt(text1 : list[list[str]], shift : int) -> list[str]:
    ''''
    decrypt data with specific shift.

    Parameters
    ----------
    text1
        List of strings in which file lines are populated.
    shift
        Shift value of character.
    
    Return
    ------
    List of string in which decrypted lines of file populated.
    '''
    decrypted_lines=[]
    text=text1[0]
    for lines in text:
        result = ""
        for char in lines:
            if char.isupper():
                result += chr((ord(char) - shift-65) % 26 + 65)
            elif char.islower():
                result += chr((ord(char) - shift-97) % 26 + 97)
            else:
                result+=char
        decrypted_lines.append(result)
    return decrypted_lines

def file_write(filename:str,text:list[str]) -> None:
    ''''
    Write content of the file.

    Parameters
    ----------
    filename
        Name of the file that needs to be encrypted.
    
    text
        list of string writes in file.
    Return
    ------
    List of string in which lines of file populated.

    '''
    with open(filename, 'w', encoding = 'Utf-8') as file:
        file.writelines(text)
