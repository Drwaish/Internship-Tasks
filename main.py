import module

def main():
    '''
    Driver code  for encryption and decryption
    Parameters
    ----------
    None

    Return
    ------
    None
    '''
    options = int(input('Enter 1 for Encryption \n 2 for decryption::'))
    try:
        if options==1:
            shift_value=input('Enter shift value')
            file_content = module.read_file(input("Enter the name of file"))
            print(file_content)
            encrypted_content = module.encrypt(file_content,shift_value)
            print('Encrypted Content',encrypted_content)
            module.file_write(input('Enter File name: '),encrypted_content)

        elif options == 2:
            shift_value = input('Enter shift value. Keep this in mind enter same value'
                        ' that entered at encryption time')
            file_content = module.read_file(input("Enter the name of file of encrypted data"))
            print(file_content)
            decrypted_content = module.decrypt(file_content,shift_value)
            print('Encrypted Content',decrypted_content)
            module.file_write(input('Enter File name for store decrypted data: '),decrypted_content)
        else:
            print('Enter Valid Options. Re-run code->Thanks')
    except Exception as exception:
        raise exception
if __name__  ==  "__main__":
    main()
