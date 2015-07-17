#coding:utf-8
#!/usr/bin/python
# Filename: cal.py
# Author: wgx
import math
from math import sin,cos
import math.pi as pi
try:
    import numpy as np
    from scipy import interpolate
    from scipy.interpolate import interp1d
except:
    print 'no numpy or scipy'
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
            if x[i] >= x[j]:
                mm += 1
        m. append(mm)
    d = m[0]
    uf = []
    for k in range(2, n+1):
        d += m[k-1]*1.0
        E = k*(k-1)/4.0
        v = k*(k-1)*(2*k+5)/72.0
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
    s = math.fabs(a) ** (-0.5)
    r = []
    for b in range(1, n+1):
        rr = 0
        for i in range(n):
            rr += (x[i]* 1.0 * func.un((n*1.0 - b*1.0)/float(a)))
        rr = rr * s
        print rr,s
        r.append(rr)
    return r
    
def sd(x, y0, y1):
    n = len(x)
    k = 0
    for i in range(n):
        if y0[i] == 0:
            continue
        k += (float(y0[i]) - float(y1[i])) ** 2 / float(y0[i]*y0[i])
    return k
    
def fm(x, y):
    n = len(x)
    xmax, ymax = [], []
    xmin, ymin = [], []
    xmax.append(x[0])
    ymax.append(y[0])
    xmin.append(x[0])
    ymin.append(y[0])
    '''
    if y[0] > y[1]:
        xmax.append(x[0])
        ymax.append(y[0])
    elif y[0] < y[1]:
        xmin.append(x[0])
        ymin.append(y[0])
    ''' 
    for i in range(1, n-1):
        if y[i] > y[i-1] and y[i] > y[i+1]:
            xmax.append(x[i])
            ymax.append(y[i])
        if y[i] < y[i-1] and y[i] < y[i+1]:
            xmin.append(x[i])
            ymin.append(y[i])
    xmax.append(x[n-1])
    ymax.append(y[n-1])
    xmin.append(x[n-1])
    ymin.append(y[n-1])
    '''        
    if y[n-1] > y[n-2]:
        xmax.append(x[n-1])
        ymax.append(y[n-1])
    elif y[n-1] < y[n-2]:
        xmin.append(x[n-1])
        ymin.append(y[n-1])
    '''
    return xmin, ymin, xmax, ymax
    
def cubic_spline_new(x, y, xnew):
    n = len(x)
    if n <= 2:
        f = interp1d(x, y, kind='slinear')
    elif n == 3:
        f = interp1d(x, y, kind='quadratic')
        '''
        import matplotlib.pyplot as plt
        #plt.plot(years,result,'r')
        plt.plot(x,y,'r',label='1')
        plt.plot(xnew,f(xnew),'b',label='2')
        plt.legend()
        plt.show()
        '''
    elif n > 3:
        return cubic_spline(x, y, xnew)
        f = interp1d(x, y, kind='cubic')
    else:
        return []
    ynew = f(xnew)
    #tck = interpolate.splrep(x, y, s=0)
    #ynew = interpolate.splev(xnew, tck, der=0)
    return ynew.tolist()
    
def cubic_spline(x, y, xnew):
    tck = interpolate.splrep(x, y, s=0)
    ynew = interpolate.splev(xnew, tck, der=0)
    return ynew
    
def anomaly(y):
    n = len(y)
    s = sum(y)
    ave = float(s) / float(n)
    for i in range(len(y)):
        y[i] -= ave
    return y
    
def getWV(y, wv='morlet', scale = 16):
    import pywt.cwt
    scales = range(2, 129, 2)
    c = pywt.cwt.cwt(y, wavelet=wv, scales=scales, data_step=1, precision=16)
    c = np.asarray(c).real
    return c[scales.index(scale)]

def getZeros(y):
    n = len(y)
    if n < 2 : return []
    p = y[0]
    r = []
    if p == 0:
        r.append(p)
    for i in range(1, len(y)):
        q = y[i]
        #print i, p , q, r
        if p*q < 0:
            k = float(q) - float(p)
            x = math.fabs(float(p))/k + i
            r.append(x)
        elif q == 0:
            r.append(i)
        p = q
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
    
    
    
def integrateSin0(trise, tset, t1, PAR1):
	temp = PAR1 * (tset-trise) / pi / sin( (t1-trise) * pi / (tset-trise) )
    result = temp *2
	if result < 0:
		return 0.0
	else:
        return result


def integrateSin1(double trise, double tset, double t1, double PAR1):
	temp = PAR1 * (tset-trise) / pi / sin( (t1-trise) * pi / (tset-trise) )
	result = temp * ( 1 - cos( (t1-trise) * pi / (tset-trise) ) )
	if result < 0.:
		return 0.0
	else :
        return result

def integrateSin2(double trise, double tset, double t2, double PAR2):
	double temp, result
	temp = PAR2 * (tset-trise) / pi / sin( (t2-trise) * pi / (tset-trise) )
	result = temp * ( cos( (t2-trise) * pi / (tset-trise)) + 1 )
	if result < 0.:
		return 0.0
	else :
        return result

def integrateXSin(double tr, double ts, double t1, double t2, double R1, double R2):
	term1 = cos(pi*(t1-tr)/(tr-ts))
	term2 = cos(pi*(t2-tr)/(tr-ts))
	term3 = 1./ sin(pi*(t2-tr)/(tr-ts))
	term4 = 1./sin(pi*(t1-tr)/(-tr+ts))
	term5 = sin(pi*(t1-tr)/(tr-ts)
	term6 = sin(pi*(t2-tr)/(tr-ts))
	
	temp1 = R2 * t1 * (tr-ts) * (term1-term2) * term3/pi/(t1-t2)
	temp2 = R1 * t2 * (tr-ts) * (term1 - term2)* term4/pi/(t1-t2)
	temp3 = (R1*(tr-ts)*term4*(-pi*t1*term1+pi*t2*term2+(tr-ts)*(term5-term6)))/(pi*pi*(t1-t2))
	temp4 = (R2*(tr-ts)*(-term3)*(-pi*t1*term1+pi*t2*term2+(tr-ts)*(term5-term6)))/(pi*pi*(t1-t2))
	
	result = temp1+temp2+temp3-temp4
	if result < 0.:
		return 0.0
	else :
        return result;

