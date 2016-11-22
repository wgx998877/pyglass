#coding:utf-8
#!/usr/bin/python
# Filename: cal.py
# Author: wgx
import math
from math import sin,cos,pi,radians
import util
import matplotlib.pyplot as plt
try:
    import numpy as np
    import scipy.stats
    from scipy import interpolate, linalg, stats
    from scipy.interpolate import interp1d
except:
    print 'no numpy or scipy'
import sklearn as skl
from sklearn import linear_model
from sklearn import svm
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
    
    
def leastsqres(x, y):
    k, b = leastsq(x,y)
    n = len(x)
    s = 0
    for i in range(n):
        ym = k*x[i]+b
        s += math.fabs(ym-y[i])
    return s
#slope, intercept, r_value, p_value, std_err
def leastsqsci(x, y):
    return scipy.stats.linregress(x, y)
    
    
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
    return round(r/float(cnt),3)
    
def rmse(x,y):
    if len(x)==0:
        return -1
    s = 0.0
    n = len(x)
    cnt = 0
    for i in xrange(n):
        if abs(y[i]-x[i]) > alp:
            continue
        s += (x[i]-y[i])**2
        cnt += 1
    s /= float(cnt)
    return round(s ** 0.5, 3)
    
def r2(x,y):
    if len(x)==0:
        return -1
    t2 = 0.0
    x2 = 0.0
    y2 = 0.0
    mx = sum(x)/len(x)
    my = sum(y)/len(y)
    tt = 0
    for i in xrange(len(x)):
        if abs(y[i]-x[i])>alp:
            continue
        t2 += (x[i]-mx)*(y[i]-my) 
        x2 += (x[i]-mx)**2
        y2 += (y[i]-my)**2
        tt += ((x[i]-mx)*(y[i]-my)-(x[i]-mx)**2)
        #print i, x[i]-mx,y[i]-my,(x[i]-mx)*(y[i]-my),(x[i]-mx)**2,tt
    #print 'r22222222222222', mx, my, t2, x2, y2 
    return round(t2**2/((x2*y2)),3)
    #return (t2/((x2*y2)**0.5))**2
    
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

def mv(y, c=12):
    ret = []
    for i in range(len(y)-int(c)):
        ret.append(sum(y[i:i+int(c)])*1.0/c)
    return ret
    
def mon2year(y):
    l = len(y)
    yn = l / 12
    r = []
    for i in range((yn)):
        r.append(0)
    m = 1
    year = 0
    for i in y:
        if m == 13:
            m = 1
            r[year] /= 12.0
            year += 1
        r[year] += i
        m += 1
    return r
    
def mon2year_ex(y ,m ,r ,delta = 10):
    n = len(r)
    ret = {}#'year':[sum,month_cnt,[(m,r)]]
    result = []
    ylist = sorted(set(y))
    for i in range((n)):
        year, month , rad = y[i], m[i], r[i]
        if year not in ret:
            ret[year] = [0,0]#[0,0,[]]
        ret[year][0] += rad
        ret[year][1] += 1
        #ret[year][2].append([(month, rad)])
    for i in ylist:
        if ret[i][1] >= delta:
            result.append([i, ret[i][0]/float(ret[i][1]), ret[i][1]])
    #print result
    return result

def removeCycle(data, cycle=12):
    m = []
    mc = []
    ma = []
    r = []
    for i in range(cycle):
        m.append(0)
        mc.append(0)
    for i in range(len(data)):
        k = i % cycle
        m[k] += data[i]
        mc[k] += 1
    for i in range(cycle):
        ma.append(float(m[i])/float(mc[i]))
    
    for i in range(len(data)):
        r.append(data[i]-ma[i%cycle])
    return r
    
def calMka(x):
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
    
def ftest(ypre, Y ,k=10,alpha=0.05):
    n = len(ypre)
    yave = float(sum(ypre))/n
    ssr, sse = 0.0, 0.0
    for i in range(n):
        ssr += (ypre[i]-yave) * (ypre[i]-yave)
        sse += (ypre[i]-Y[i]) * (ypre[i]-Y[i])
    f = (ssr / k) / (sse / (n-k-1))
    return f
'''
    p_value = scipy.stats.f.cdf(scipy.stats.F, ypre, Y)
    return p_value
'''
    
def pearson(x, y):
    return scipy.stats.pearsonr(x, y)
    
def spearman(x, y):
    return scipy.stats.spearmanr(x, y)
    
def line_regr(x,y,return_obj=True,fit_intercept=False):
    #regr = linear_model.LinearRegression(copy_X=True,n_jobs=1)
    regr = linear_model.LinearRegression(copy_X=True,fit_intercept=fit_intercept,n_jobs=1)
    regr.fit(x,y)
    if return_obj:
        '''
        regr.fit # train
        regr.predict # Xlist
        regr.score # variance score x,y
        '''
        return regr
        
    else:
        return regr.coef_, regr.intercept_
        
def line_regr2(x,y,fit_intercept=False):
    regr = linear_model.LarsCV(copy_X=True,fit_intercept=fit_intercept,n_jobs=1)
    regr.fit(x,y)
    return regr
    
def poly2_regr(x, y):
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    poly.fit(x, y)
    return poly
        
    
def pls_regr(x, y):
    from sklearn.cross_decomposition import PLSRegression
    n = len(x[0])
    if n < 2:
        raise TypeError
    score = -999999999999
    pls = None
    '''
    for i in range(3, n):
        pls2 = PLSRegression(n_components=i)
        pls2.fit(x,y)
        cscore = pls2.score(x, y)
        #print i, cscore 
        if cscore > score:
            pls = pls2
            score = cscore
    '''
    pls = PLSRegression(n_components=5)
    pls.fit(x,y)
    return pls

def ridge_regr(x,y,alpha=0.1):
    if type(alpha) == list:
        #clf = linear_model.RidgeCV(alphas=alpha, normalize=True)
        clf = linear_model.RidgeCV(alphas=alpha, fit_intercept=False)
        clf.fit(x,y)
        #alpha = clf.alpha_
        #if alpha != 0.1:
            #print alpha,'*',
        return clf
    regr = linear_model.Ridge(copy_X=True, alpha=alpha)
    regr.fit(x,y)
    return regr
        
def svr_regr(x,y):
    #regr = svm.SVR(kernel='linear')
    #regr = svm.SVR(kernel='precomputed')
    #regr = svm.NuSVR()
    regr = svm.LinearSVR(epsilon=0)
    regr.fit(x,y)
    return regr

from sklearn import tree, ensemble    
def per_regr(x,y):
    regr = tree.DecisionTreeRegressor()
    regr.fit(x,y)
    return regr
    
def ada_regr(x,y):
    regr = ensemble.AdaBoostRegressor()
    regr.fit(x,y)
    return regr
    
def rfr_regr(x,y):
    regr = ensemble.RandomForestRegressor()
    regr.fit(x,y)
    return regr
    
    
from pyearth import Earth
def mars_regr(x, y):
    model = Earth()
    regr= model.fit(np.asarray(x),np.asarray(y))
    return regr
        
def grnn_regr(x,y,std=0.1):
    from neupy.algorithms import GRNN
    regr = GRNN(std = std, verbose=False)
    regr.train(x,y)
    return regr
    
def logis_regr(x,y,return_obj=True,cs = True):
    if cs:
        #regr = linear_model.LogisticRegressionCV()
        regr = linear_model.LogisticRegressionCV(solver='liblinear')
    else:
        regr = linear_model.LogisticRegression(C=10.0)
    try:
        regr.fit(x,y)
    except:
        regr = line_regr(x, y)
    if return_obj:
        return regr
    else:
        return regr.coef_, regr.intercept_
    
def mka(x):
    xr = x[::-1]
    ua, ub = calMka(x)[0], calMka(xr)[0]
    return ua, ub

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
    
def guiyi(y):
    n = len(y)
    s = sum(y)
    ma = float(max(y))
    mi = float(min(y))
    r = []
    ave = float(s) / float(n)
    for i in range(len(y)):
        ret = (y[i]-mi) / (ma-mi)
        r.append(ret)
    return r
    
def deseason(y, st=1900):
    import pandas as pd
    import statsmodels.api as sm
    ye = '%d-%02d' % (st + len(y)/12, len(y)%12 + 1)
    m = pd.date_range('%04d-01' % st, ye, freq='M')
    yd = pd.Series(y,m)
    res = sm.tsa.seasonal_decompose(yd)
    return res.trend[6:-6]
    
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
   
def pca(data, topNfeat=999999):
    meanV = np.mean(data,axis=0)
    meanR = data - meanV
    covM = np.cov(meanR,rowvar=0)
    eVal,eVec = np.linalg.eig(mat(covM))
    eValInd = np.argsort(eVal)
    eValInd = eValInd[:-(topNfeat+1):-1]
    redEVec = eVec[:,eValInd]
    lowData = meanR * redEVec
    reconMat = (lowData * redEVec.T) + meanV
    return lowData, reconMat
    
    
def integrateSin0(trise, tset, t1, PAR1):
    temp = PAR1 * (tset-trise) / pi / sin( (t1-trise) * pi / (tset-trise) )
    result = temp *2
    if result < 0:
        return 0.0
    else:
        return result


def integrateSin1(trise, tset, t1, PAR1):
    temp = PAR1 * (tset-trise) / pi / sin( (t1-trise) * pi / (tset-trise) )
    result = temp * ( 1 - cos( (t1-trise) * pi / (tset-trise) ) )
    if result < 0.:
        return 0.0
    else :
        return result

def integrateSin2(trise, tset, t2, PAR2):
    temp = PAR2 * (tset-trise) / pi / sin( (t2-trise) * pi / (tset-trise) )
    result = temp * ( cos( (t2-trise) * pi / (tset-trise)) + 1 )
    if result < 0.:
        return 0.0
    else:
        return result
        
def integrateXSin(tr, ts, t1, t2, R1, R2):
    term1 = cos(pi*(t1-tr)/(tr-ts))
    term2 = cos(pi*(t2-tr)/(tr-ts))
    term3 = 1.0/sin(pi*(t2-tr)/(tr-ts))
    term4 = 1.0/sin(pi*(t1-tr)/(-tr+ts))
    term5 = sin(pi*(t1-tr)/(tr-ts))
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


def ApEn(x, m=2, r=-1):
    if len(x) == 0:
        return None
    x = np.asarray(x)
    if r == -1:
        r = 0.15 * np.std(x)
    return calAE(x, m, r) - calAE(x, m+1, r)

def calAE(x, m, r):
    n = len(x)
    tn = n - m + 1
    ux = np.zeros((tn, m))
    for i in xrange(tn):
        for j in range(m):
            ux[i, j] = x[i + j]
    d = np.zeros((tn, tn))
    c = np.zeros(tn)
    lnc = np.zeros(tn)
    ret = 0
    
    for i in xrange(tn):
        for j in xrange(tn):
            for k in range(m):
                tmp = np.fabs(ux[i, k] - ux[j, k])
                if d[i, j] < tmp:
                    d[i, j] = tmp
                    
    for i in xrange(tn):
        t = 0.0
        for j in xrange(tn):
            if d[i, j] <= r:
                t += 1.0
        c[i] = float(t) / float(tn)
        lnc[i] = math.log(c[i])
        ret += lnc[i]
    ret = float(ret) * 1.0 / float(tn)
    return ret
    
def mt(x, slen):
    n = len(x)
    tn = n - slen * 2
    t = []
    for i in range(tn):
        x1 = x[i:i+slen]
        x2 = x[i+slen:i+slen*2]
        x1ave = np.mean(x1)
        x2ave = np.mean(x2)
        x1var = np.var(x1)
        x2var = np.var(x2)
        s = np.sqrt((float(slen) * x1var + float(slen) * x2var) / float(slen+slen-2))
        tmp = (x1ave - x2ave) / (s * np.sqrt(2.0/float(slen)))
        t.append(tmp)
    return t
def mean(l):
    if len(l) == 0:
        return 0
    return float(sum(l))/float(len(l))
    
def calCV(x):
    l = len(x)
    if l == 0:
        return 0
    m = sum(x) *1.0 / float(l)
    if m == 0:
        return 0
    s = 0.0
    for i in x:
        s += (float(i) - float(m)) ** 2
    s = s * 1.0 / float(l)
    s = s ** 0.5
    return s * 1.0 / float(m)

def TestApEnMt():
    n = 2000
    x1 = range(1, n+1)
    y1 = []
    y1a = []
    for i in range(n/2):
        y1.append(math.sin(0.2*x1[i]) * 2 + 1)
    for i in range(n/2):
        y1.append(math.sin(0.2*x1[i+1000]) * 1.5 + math.cos(0.5*x1[i+1000])*2 -0.5)
    plt.plot(x1, y1)
    plt.show()
    y1a = mt(y1, 100)
    plt.plot(x1[:1800], y1a)
    plt.show()
    y1a = []
    for i in range(n-200):
        tmpx = y1[i:i+200]
        yt = ApEn(tmpx)
        print i, yt
        y1a.append(yt)
    print len(y1a), len(x1[0:1800])
    plt.plot(x1[0:1800], y1a)
    plt.show()
    
def getESS(p, r, use_pcc = True, use_rss = False):
    k = []
    if use_pcc:
        all = []
        for i in range(len(r)):
            tmp = []
            for j in range(len(p)):
                tmp.append(p[j][i])
            tmp.append(r[i])
            all.append(tmp)
        pc, pv = partial_corr(all)
    for i in range(len(p)):
        #pcr.append((pc[i][-1], pv[i][-1]))
        if use_pcc:
            ipc, ipv = pc[i][-1], pv[i][-1]
        else:
            ipc, ipv = pearson(p[i], r)
        #print i, ipc, ipv, pearson(p[i], r)
        k.append([i, abs(ipc), ipc, ipv, 0, p[i]])
    k = sorted(k, key=lambda s : -s[1])
    lEss = 0
    x = []
    cnt = 0
    for i in k:
        cx = i[5]
        x.append(cx)
        cEss = ess(x, r)
        cnt += 1
        i[4] = cEss - lEss
        del i[5]
        lEss = cEss
    if use_rss:
        Rss = rss(x, r)
        k.append([cnt, 0, 0, 0, Rss])
        lEss += Rss
    for i in k:
        i[4] = i[4] * 100 / lEss
        del i[1]
    k = sorted(k, key=lambda s : s[0])
    #id, cor, pvalue, es
    return k

def ess(x, y):
    if len(y) == 0:return 0
    xt = []
    for i in range(len(x[0])):
        tmp = []
        for j in range(len(x)):
            tmp.append(x[j][i])
        xt.append(tmp)
    yt = line_regr(xt, y, fit_intercept = True).predict(xt)
    my = sum(y)/len(y)
    es = 0
    for i in range(len(y)):
        es += (yt[i]-my)*(yt[i]-my)
    return es
    
def rss(x, y):
    if len(y) == 0:return 0
    xt = []
    for i in range(len(x[0])):
        tmp = []
        for j in range(len(x)):
            tmp.append(x[j][i])
        xt.append(tmp)
    yt = line_regr(xt, y, fit_intercept = True).predict(xt)
    my = sum(y)/len(y)
    rs = 0
    for i in range(len(y)):
        rs += (yt[i]-y[i])*(yt[i]-y[i])
    return rs
    
def resize(d, x=180.0, y=360.0):
    l1, l2 = d.shape
    dt = scipy.ndimage.interpolation.zoom(d, (180.0/l1,360.0/l2))
    return dt
    
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of variables in C, controlling 
    for the remaining variables in C.
    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable
    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    Pv = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot(beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr, v = stats.pearsonr(res_i, res_j)
            P_corr[i, j] = corr
            P_corr[j, i] = corr
            Pv[i, j] = v
            Pv[j, i] = v
    #print Pv
    return P_corr, Pv
