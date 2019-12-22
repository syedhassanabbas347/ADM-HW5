#with open(r'C:\Users\aless\Desktop\HW5_ADM\data', 'r') as f:
 #   arr = list(map(int, (f.read().split())))

import pandas as pd
import numpy as np
import time

def func_2():
    sol = 'Da Fare'
    return sol


import func_1 as f1
import func_3 as f3
import func_4 as f4




if __name__ == "__main__":

    print("\nChoose between the following functionalities:")
    print("1 - Find the Neighbours!")
    print("2 - Find the smartest Network!")
    print("3 - Shortest Ordered Route")
    print("4 - Shortest Route")
    choice = int(input("Enter your choice: "))
    while choice not in [1, 2, 3, 4]:
        print("Please, insert a valid choice!")
        choice = int(input("Enter your choice: "))

    if choice == 1:
        sol = f1.func_1()


    if choice == 2:
        sol = func_2()

    if choice == 3:
        sol = f3.func_3()

    if choice == 4:
        sol = f4.func_4()
