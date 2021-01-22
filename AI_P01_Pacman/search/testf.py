

def digui(list,depth):
    if(len(list)>5):
        return list
    list.append(depth)
    l = digui(list,depth+1)
    return l

list = digui([],0)
print list