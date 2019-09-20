import os

def mkdir(path):
    """
    mkdir of the path
    :param input: string of the path
    return: boolean
    """
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        print(path+' is created!')
        return True
    else:
        print(path+' already exists!')
        return False