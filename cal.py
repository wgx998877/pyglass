#coding:utf-8
#!/usr/bin/python
# Filename: cal.py
# Author: wgx
#import numpy as np
import math
alp = 10000

def leastsq(x,y):
    """
    x,y分别是要拟合的数据的自变量列表和因变量列表
    """
    meanx = sum(x) / len(x)   #求x的平均值
    meany = sum(y) / len(y)   #求y的平均值

    xsum = 0.0
    ysum = 0.0

    for i in xrange(len(x)):
        xsum += (x[i] - meanx)*(y[i]-meany)
        ysum += (x[i] - meanx)**2

    k = xsum/ysum
    b = meany - k*meanx

    return k,b   #返回拟合的两个参数值
def bias(x,y,k=1,b=1):
    if len(x)==0:
	return -1
    r = 0.0
    n = len(x)
    cnt = 0
    for i in xrange(n):
	if abs(y[i]-x[i])>alp:
	    continue
        cnt += 1
        r += (y[i]-x[i])
    return r/float(cnt)
    
def rmse(x,y):
    if len(x)==0:
	return -1
    s = 0.0
    n = len(x)
    cnt = 0
    for i in xrange(n):
	if abs(y[i]-x[i])>alp:
	    continue
        s += (x[i]-y[i])**2
        cnt += 1 
    s/=float(cnt)
    return s ** 0.5
def r2(x,y):
    if len(x)==0:
	return -1
    t2 = 0.0
    x2 = 0.0
    y2 = 0.0
    mx = sum(x)/len(x)
    my = sum(y)/len(y)
    for i in xrange(len(x)):
	if abs(y[i]-x[i])>alp:
	    continue
        t2 += (x[i]-mx)*(y[i]-my) 
        x2 += (x[i]-mx)**2
        y2 += (y[i]-my)**2
    return (t2/((x2*y2)**0.5))**2
    
def mk(x):
    n = len(x)
    S = 0
    for i in xrange(0, n):
        for j in xrange(i+1, n):
            sgn = 0
            if x[j] > x[i]: sgn = 1
            elif x[j] == x[i]: sgn = 0
            else : sgn = -1
            S += sgn
    Zc = 0
    if S > 0:
        Zc = (S - 1)/((n*(n-1)*(2*n+5))**0.5)
    elif S < 0:
        Zc = (S + 1)/((n*(n-1)*(2*n+5))**0.5)
    else :
        Zc = 0
    return Zc

def mka(x):
    n = len(x)
    m = []
    for i in range(n):
        mm = 0
        for j in range(i):
            if x[i] > x[j]:
                mm += 1
        m. append(mm)
    d = m[0]
    uf = []
    for k in range(2, n+1):
        d += m[k-1]
        E = k*(k-1)/4
        v = k*(k-1)*(2*k+5)/72
        ufk = (d - E) / (v ** 0.5)
        uf.append(ufk)
    ub = uf[::-1]
    return uf ,ub
    
def wavelet(t='haar', x=[], a=2):
    class haar:
        def u(self, t):
            if t >= 0 and t < 0.5:
                return 1.0
            elif t >= 0.5 and t < 1:
                return -1.0
            else :
                return 0
        def un(self, t):
            return self.u(t)
    class marr:
        a = (2.0/(((math.pi**0.5)*3.0)**0.5))
        def u(self, t):
            return a*(1-t*t)*math.exp((-t*t/2.0))
            
        def un(self, t):
            return self.u(t)
    class morlet:
        def u(self, t, w0=6):
            return math.exp(-t*t/2.0) * (math.pi ** -0.25) * math.cos(w0*t)#(complex(math.cos(w0*t), math.sin(w0*t)))
        def un(self, t, w0=6):
            return math.exp(-t*t/2.0) * (math.pi ** -0.25) * math.cos(w0*t)#(complex(math.cos(w0*t), math.sin(w0*t)))
    class gauss:
        def u(self, t):
            return -(t*math.exp(-t*t/2.0))/((2.0*math.pi)**0.5)
        def un(self, t):
            return self.u(t)
    n = len(x)
    if t == 'haar':
        func = haar()
    elif t == 'marr':
        func = marr()
    elif t == 'morlet':
        func = morlet()
    elif t == 'gauss':
        func = gauss()
    else :
        raise NameError("%s wavelet not set" % t)
    s = fabs(a) ** (-0.5)
    r = []
    for b in range(1, n+1):
        rr = 0
        for i in range(n):
            rr += (x[i] * func.un(float(n - b)/float(a)))
        rr = rr * s
        r.append(rr)
    return r
    
    
def RS(x):
    n = len(x)
    if n <= 0 : return
    RS = []
    XS = []
    for t in xrange(2,n+1):
        es = sum(x[:t])*1.0/t
        st = 0.0
        for j in xrange(t):
            st += (x[j] - es)**2
        st = (st*1.0/t)**0.5
        X = []
        s = 0
        for j in xrange(t):
            s+=x[j]
            X.append((s-(j+1)*es))
        rt = max(X[:t])*1.0 - min(X[:t])*1.0
        RS.append(math.log(rt*1.0/st))      
        XS.append(math.log(t))
    return leastsq(XS, RS)
   
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
