""" 
Numbered the inout code.

"""


def remove_first(lines: str) -> str:
    """
    Remove useless character.

    Parameters
    ----------
    lines
        String that contain use less character.

    Return
    ------
    Filtered String .
    """
    var = ' '
    if lines[0] == '\\' and lines[1] == 'n':
        for i in range(2, len(lines)):
            var += lines[i]
    elif lines[0] == "'":
        for i in range(1, len(lines)):
            var += lines[i]
    return var


def remove_doubleslash(lines):
    """
    Remove Double Slash
    """
    for i in range(len(lines)-2):
        if lines[i] == "\\" and lines[i+1] == "\\":
            lines.replace(lines[i], "")
    return lines


def check_new_line(character: str) -> bool:
    """
    Check and filter new line character.

    Parameters
    ----------
    character
        Character to filter.
    Return
    ------
    Bool
    """
    if character.isdigit():
        return True

    elif character.isalpha() and character != '\n':
        return True
    elif character in ['[', ']', '{', '}', '(', ')', '"', ":"]:
        return True
    else:
        return False


def prepare_string(code_string: str) -> list[str]:
    """
    Prepare list of string for manipulation.

    Parameters
    ----------
    code_string
        Contain string of code.

    Return
    ------
    List of string.
    """
    # mad = repr(STR)
    mad = repr(code_string)
    # print(mad.lstrip())
    mad1 = mad.split('\n')
    # print(mad1[0])
    code = []
    var: list[str] = []
    for i in range(0, len(mad1[0])-2):
        # print(mad1[0][i])
        if mad1[0][i] == '\\' and mad1[0][i+1] == 'n' and not check_new_line(mad1[0][i+2]):
            code.append(''.join(var))
            var.clear()
        var.append(mad1[0][i])
    # print("Code in pS",code)
    return code


def append_number(code: list[str]) -> str:
    """
    Append numner to code.

    Parameters
    ----------
    code
        List of string contain processed string.

    Return
    ------
    Strin contain numbered code.
    """
    print("Code:", code)
    preocessed_code = []
    i = 1
    var = ""
    for lines in code:
        var = remove_first(lines)
        var = remove_doubleslash(var)
        print(var)
        if var == " " and lines != '\\n':
            # Append line number before every line
            preocessed_code.append(str(i) + ': ' + lines)
        else:
            # Append line number before every line
            preocessed_code.append(str(i) + ': ' + var)

        i += 1
    numberd_code = '\n'.join(preocessed_code)
    return numberd_code
