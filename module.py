"""
Module to manipulate string
"""
STR ="""def print(s):
    print(f"this is your string\n{s}")"""
def to_raw(string_of_code: str)->str:
    '''
    Convert simple string into raw string

    Parameters
    ----------
    string
        A string contain code
    
    Return
    ------
    String, converted into raw string
    '''
    return fr"{string_of_code}"
def code_input()->str:
    '''
    Multiline input of code

    Parameters
    ---------
    None

    Return
    -----
    str
      String contain input for manipulating
    '''
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z to end input")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    code_line = "\n".join(contents)
    return code_line

def split_lines(code_string: str)->list[str]:
    '''
    Split string base on end of line

    Parameters
    ---------
    code_string
       String contain data to split
    
    Return
    ------
    list[str]
      Contain data after split base on end of line (\n) 
    '''
    raw_string=to_raw(code_string)
    print(raw_string)
    return raw_string.split('\n')


def append_line_before_line(code:str):
    """
    Append line numbers before code

    Parameters
    ----------
    code
        Data stored in this variable

    Return
    ------
    str
        appending with line numbers 
    """
    lines_of_code=split_lines(STR)
    i=1
    preocessed_code=[]
    for lines in lines_of_code:
        preocessed_code.append(str(i)+': '+lines) #Append line number before every line
        i+=1
    numberd_code='\n'.join(preocessed_code)
    return numberd_code
    