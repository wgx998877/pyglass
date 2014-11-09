#coding:utf-8
import numpy as np

def leastsq(x,y):
    meanx = sum(x)/len(x)
    meany = sum(y)/len(y)

    xsum = 0.0
    ysum = 0.0
    for i in xrange(len(x)):
        xsum += (x[i] - meanx) * (y[i] - meany)
        ysum += (x[i] - meanx) ** 2
    k = xsum/ysum
    b = meany - k*meanx
    return k,b
def calrmse(x,y):
    s = 0.0
    for i in xrange(len(x)):
        s += (x[i] - y[i])**2
    s/=len(x)
    return s**0.5
def calr2(x,y):
    t = 0.0
    c = 0.0
    mx = sum(x)/len(x)
    for i in xrange(len(x)):
        t += (x[i]-mx) ** 2
        c += (x[i]-y[i]) ** 2
    return (t-c)*1.0/t*1.0
def calbias(x,y,k,b):
    r = 0
    for i in xrange(len(x)):
        r += abs(y[i]-(k*x[i]+b))
    return r/len(x)

def pca(data,topNfeat=999999):
    meanV = np.mean(data,axis=0)
    meanR = data - meanV
    covM = np.cov(meanR,rowvar=0)
    eVal,eVec = np.linalg.eig(mat(covM))
    eValInd = np.argsort(eVal)
    eValInd = eValInd[:-(topNfeat+1):-1]
    redEVec = eVec[:,eValInd]
    lowData = meanR * redEVec
    reconMat = (lowData * redEVec.T) + meanV
    return lowData,reconMat
