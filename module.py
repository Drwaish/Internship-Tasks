"""Methods of Contact Management System """


import json


def write_json(contacts: dict) -> None:
    """
    Read json file in which our contacts are stored

    Parameters
    ----------
    Dictionary of contacts

    Return
    ------
    None
    """
    try:
        json_object = json.dumps(contacts, indent = 4)
        with open('contacts.json', 'w', encoding = 'Utf-8') as file:
            file.write(json_object)

    except ReferenceError:
        print('File Writing Error')


def read_json() -> dict:
    """
    Read json file in which our contacts are stored

    Parameters
    ----------
    None

    Return
    ------
    Dictionary of contacts
    """
    try:
        with open('contacts.json', 'r', encoding = 'Utf-8') as file:
            dict_contact = json.load(file)
    except ReferenceError:
        print('Error in file reading')
    return dict_contact


def add_contacts(key: str, value: str, contacts: dict) -> None:
    """
    Add contacts of user in management system.

    Parameters
    ----------
    Key
        From key we access the contacts
    Value
        Store Contacts of manageemnt sysem
    Return
    ------
    None
    """
    contacts[key] = value


def delet_contacts_key(key: str, contacts: dict) -> None:
    """
    Delete contacts of user in management system.

    Parameters
    ----------
    Key
        From key we access the contacts to delete.
    contacts
        Dictionary in which our contacts are present

    Return
    ------
    None
    """
    try:
        var = False  #flag value to determine whether key or value found or notuu
        for elements in contacts:
            if elements == key:
                # contacts.pop(elements)
                var = True
        if var:
            contacts.pop(key)
        if not var:
            print("Key not found")
    except RuntimeError:
        print("Enter Valid Key. Thanks")


def delet_contacts_value(value: str, contacts: dict) -> None:
    """
    Delete contacts of user in management system.

    Parameters
    ----------
    value
        From value we access the contacts to delete.
    contacts
        Dictionary in which our contacts are present

    Return
    ------
    None
    """
    try:
        var = False #flag value to determine whether key or value found or not
        key = 'None'
        for elements in contacts:
            if contacts[elements] == value:
                # contacts.pop(elements)
                key = elements
                var = True
        if var:
            contacts.pop(key)
        if not var:
            print("Value not found")
    except RuntimeError:
        print("Enter Valid Key. Thanks")


def view_contacts(contacts: dict) -> None:
    """
    View contacts of user in management system.

    Parameters
    ----------
    contacts
        Dictionary in which our contacts are present

    Return
    ------
    Dictionary in which all our contacts are available
    """
    print("Here is you contacts: \n", contacts)


def update_contacts_key(old_key: str, new_key: str, contacts: dict) -> None:
    """
    Update contacts of user in management system.

    Parameters
    ----------
    Key
        From key we access the contacts
    Value
        Store Contacts of managemnt sysem
    Return
    ------
    None
    """
    try:
        var = False  #flag value to determine whether key or value found or not
        for elements in contacts:
            if elements == old_key:
                value = contacts[old_key]
                contacts.pop(old_key)
                contacts[new_key] = value
                var = True
        if not var:
            print("Key not found")
    except RuntimeError:
        print('Problem with Contact file ')


def update_contacts_value(old_value: str, new_value: str, contacts: dict) -> None:
    """
    Update contacts of user in management system.

    Parameters
    ----------
    old_value
        From value we access the contacts
    new_value
        Change ole value with new value
    Return
    ------
    None
    """
    try:
        var = False #flag value to determine whether key or value found or not
        for elements in contacts:
            if contacts[elements] == old_value:
                contacts[elements] = new_value
                var = True
        if not var:
            print("Value  not Found")

    except RuntimeError:
        print('Problem with Contact file ')
