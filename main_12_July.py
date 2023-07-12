import core


my_num=[1,2,3,4,5]
def main():
    #Generator and Yield
    my_square=core.generator()
    print(core.generator. __doc__)
    print(id(next(my_square)))
    print(id(next(my_square)))
    for i in my_square:
        print(i)
    #Working with *args
    core.argss('Hi','Zain')
    print(core.argss. __doc__)

    #use of lambda
    my_doubler=core.my_func(2)  
    print(core.my_func. __doc__)
    print(my_doubler(11))

    #run iterators
    core.print_iterator(my_num)
    print(core.print_iterator. __doc__)
    print(core.print_iterator.__annotations__)

if __name__=='__main__':
    main()

#decorator function create logs
core.display_func('Zain',23)  

#named tuple using collections
core.named_tuple()
