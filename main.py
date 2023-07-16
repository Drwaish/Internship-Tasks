import module


def main() -> None:
    """
    Its a contact management system with for diiferent menues
    User can add contacts
    User can View contacts
    User can delete contacts
    User can Update contacts

    Parameters
    ----------
    None

    Return
    ------
    None

    """
    contacts = module.read_json()
    try:

        while True:
            print('''Press 1 for add contacts.
                     Press 2 for view  contacts
                     Press 3 for delete  contacts
                     Press 4 for update contacts
                     ''')
            options = int(input("Select from Menu: "))
            if options == 1:
                name = input("Enter the name : ")
                contact_number = input(
                    "Enter Contact number : ")  # use as a key
                module.add_contacts(key = contact_number,
                                    value = name, contacts = contacts)
            elif options == 2:
                module.view_contacts(contacts)

            elif options == 3:
                try:
                    while True:
                        print('''
                             Press k for delete using contact number : 
                             Press v for delete using name : 
                             ''')
                        opt_inner = input("Enter option : ")
                        if opt_inner == 'k':
                            key = input('Enter the contact number : ')
                            module.delet_contacts_key(key, contacts)
                        elif opt_inner == 'v':
                            value = input('Enter the name : ')
                            module.delet_contacts_value(value, contacts)
                        else:
                            print("Enter valid options")
                except EOFError:
                    print('Successfully Deleted')
            elif options == 4:
                try:
                    while True:
                        print('''Press k for update  contact number:
                                 Press v for update  name :
                                 ''')
                        opt_inner = input("Enter option : ")
                        if opt_inner == 'k':
                            old_contact_number = input(
                                "Enter old Contact number: ")
                            new_contact_number = input(
                                "Enter new Conttact number: ")
                            module.update_contacts_key(old_contact_number,
                                                       new_contact_number, contacts)
                        elif opt_inner == 'v':
                            number = input("Enter old name: ")
                            value = input("Enter new name: ")
                            module.update_contacts_value(
                                number, value, contacts)
                        else:
                            print("Enter valid value")
                except EOFError:
                    print("All updates are done\n")
    except EOFError:
        module.write_json(contacts)
        print('Bye')


if __name__ == "__main__":
    main()
