# List a = [2,6,3,9] n=15  
# Write an algorithm which returns the indices of the numbers from list ‘a’ 
# such that they add up to ‘n’. Example: In this case 6 + 9 = 15 so the function should 
# return [1,3] because the index of 6 is 1 and the index of 9 is 3. 
# 

flag=0
list_item=[]
for index,elem in enumerate(a):
    for index2,elem2 in enumerate(a):
        if elem+elem2==15:
   
            print(index,index2)
