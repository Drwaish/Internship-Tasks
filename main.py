import module
def main():
    """
    Driver code for module.py
    
    Parameters
    ----------
    None

    Return
    ------
    None
    """
    code_string=module.code_input()
    numbered_code=module.append_line_before_line(code_string)
    print(numbered_code)
    print(type(numbered_code))
    
if __name__=='__main__':
    main()
