import logging
from collections import namedtuple
from typing import Callable,Type

def generator():
    '''
    A generator to square of number from 1 to 5
    
    Parameters
    ----------
        None
    
    Yield
    -----
        int

    '''
    for i in range(1, 6):
        yield i*i


def argss(*args):
    '''
    Print data in *args 
    
    Parameters
    ----------
        *args
    
    Return
    ------
        None

    '''
    for element in args:
        print(element)


def my_func(num: Callable[[int],int]) :
    '''lambda function multiply with parameters n
    
    Parameters
    ----------
    num: int
        num multiply with a
    
    Return
    ------
    int

    '''
    return lambda a: a * num


def print_iterator(arr: list):
    '''
    Elements in arr print using iter

    Parameters
    ----------
    arr:list
        contain elements for printing
    
    Return
    ------
    None
    '''
    myitr = iter(arr)
    for element in myitr:
        print(element)


def my_logger(origin_func):
    """
    Create log with name and age

    Parameters
    ----------
    origin_func: Function
        A wrapper function who call inside decorated function

    Return
    ------
    Method
    Method that run the wrapper function
    """
    logging.basicConfig(filename='{}.log'.format(
        origin_func.__name__), level=logging.INFO)

    def wrapper(*args, **kwargs):
        logging.info('Ran with args:{args}, and kwargs: {kwargs} ')
        return origin_func(*args, **kwargs)
    return wrapper


@my_logger
def display_func(name, age):
    """
    Display name and age 

   Parameters
    ----------
    name: str
        name of user.
    age: int
        age of user
    Returns
    -------
    None
    """
    print(f'display_info ran with arguments({name},{age})')


def named_tuple():
    """
    Assign color to value using namedtuple
    Parameters
    ----------
    None
    
    Return
    ------
    None

    """
    Color = namedtuple('Color', ['red', 'green', 'blue'])
    color = Color(55, 155, 255)
    print(color)

# def return_multiple_using_dict(name: Union[str,int], age: Union[str,int], message: Union[str,int])-> dict:
def return_multiple_using_dict(name: Type, age: Type, message: Type)-> dict:

    """
    This function take  a name as str and age as int and
    message as str and return these values as dict
    Parameters
    ----------
    name: str
        name of the user
    age: int
        age of the user
    message: str
        message for the user
    Return
    dict
        return a dictionary contianing name of user ,age of user and 
        message for user 
    """
    user_dict= dict()
    user_dict['name']= name
    user_dict['age']= age
    user_dict['message']= message
    return user_dict
