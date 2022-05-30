def transform(num):
        h = num // 3600
        m = num % 3600 //60
        s = num % 3600 % 60
        if h == 0:
            h = ''
        else:
            h = str(int(h)) + 'h'
        if m == 0:
            m = ''
        else:
            m = str(int(m)) + 'm'
        if s == 0:
            s =''
        else:
            s = str(int(s)) + 's'
        return h+m+s

def saveDict(dict, path):
    if isinstance(dict, str):
        dict = eval(dict)
    with open(path, 'w') as f:
        f.write(str(dict))  # dict to str

def loadDict(path):
    with open(path, 'r') as f:
        dict = eval(f.read())  # eval
    return dict


