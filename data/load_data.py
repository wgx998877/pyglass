def loadDataSet(filename,delim='\t'):
    fr = open(filename)
    strArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in strArr]
    return mat(datArr)

def loadCsv(filename):
    return loadDataSet(filename,',')
