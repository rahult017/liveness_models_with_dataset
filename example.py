IT= "my name is Rahul"
# o1 = "Ym Eman Si Luhar"
# o2 = "Y1 M2 E1 A2 N1 S1 I1 L1 U1 H1 R1"

o1 =' '.join(string[::-1].title() for string in IT.split())
print(o1)
title_str = o1.upper()
char_count ={}
o2 =" "
for char in title_str:  # Iterate over characters in `o1`
    if char != " ":  # Ignore spaces
        char_count[char] = char_count.get(char, 0) + 1
        

for char,count in char_count.items():
    o2+= f"{char}{count} "
print("o2 =", o2)


l1 = [2,3,2,1,5,6,7,8,9,5,6]
# # o1 = [2,5,6]
# # o2 = [3,1,7,8,9]
dict_ele = {}
o3 =[]
o4 =[]
for element in l1:
   dict_ele[element] = dict_ele.get(element, 0) + 1

for number,count in dict_ele.items():
   if count > 1:
      o3.append(number)
   else:
      o4.append(number)
print(o3)
print(o4)

list1 = [7,1,5,2,6]
max_profit = 0 
buy_price = float("inf")
for price in list1:
   buy_price = min(buy_price,price)
   current_profit = price - buy_price
   max_profit = max(current_profit,max_profit)
print(max_profit)

def fib(n):
    if n <=0:
        return 1
    else:
        a,b = 0,1
        for _ in range(0,n+1):
            a,b = b,a+b
            yield a

num = [num for num in fib(5)]
print(num)

def check_string(func):
    def wrapper(string1,string2):
        string1*= 5
        return func(string1,string2)
    return wrapper

@check_string
def valid(string1,string2):
    return string1 + string2
print(valid(9,5))