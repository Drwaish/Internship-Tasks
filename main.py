import module


STR = """ def (s)\n:
     print(f"this is your string\n{s}")
     \nprint(f"\nthis is your string\n{s}\n")
     """
# STR="""
    
#     """

def main():
    """
    Driver code for append number in string"

    Parameters
    ----------
    None

    Return
    ------
    None

    """
    code = module.prepare_string(STR)
    print(module.append_number(code))


if __name__ == "__main__":
    main()
