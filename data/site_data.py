# author : wgx
import os
dpath = os.path.split(os.path.realpath(__file__))[0]
class site:
    def __init__(self,id='',name='',pro='',rank='',lat=0,lon=0,alt='0',time='',data=[]):
        if rank == '':
            rank = 3
        self.name = name
        self.id = id
        self.pro = pro
        self.rank = int(rank)
        self.lat=float(int(lat)/100)+(int(lat)%100)/60.0
        self.lon=float(int(lon)/100)+(int(lon)%100)/60.0
        self.alt=float(int(alt.replace('(','').replace(')','')))/10.0
        self.time=time
        self.data=data
    def show(self):
        print "%s,%s,%s,%s\n" % (self.id,self.name,self.pro,self.data)
        for i in self.data:
            print i
def get_site_info():
    s_info = open(dpath+'/site_txt/site_info.txt')
    sites = []
    for i in s_info:
        try:
            i = i.split(',')
            s = site(i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7].strip())
            sites.append(s)
        except :
            print 'error:'
            print i
    s_info.close()
    return sites
def get_daily_data(year=0,last=0):
    sites = get_site_info()
    l = len(sites)
    f = open(dpath+'/site_txt/site_daily_data.txt')
    j = 0
    for i in f:
        if ',' not in i:
            continue
        i = i.strip().split(',')
        y = int(i[1])
        time = ("%4d%02d%02d" % (int(i[1]),int(i[2]),int(i[3])))
        tmp = [time,i[4],i[5],i[6],i[7],i[8],i[9]]
        if year==0 or (year+last<=y) :
            for x in xrange(0,l):
                if sites[x].id == i[0]:
                    #j = x
                    sites[x].data= sites[x].data + [tmp]
                    break
    f.close()
    for i in sites:
        if i.data==[]:
            sites.remove(i)
    print "get %d sites data!\n" % len(sites)
    return sites
def get_month_data(year=0,last=0):
    sites = get_site_info()
    f = open(dpath+'/site_txt/site_month_data.txt')
    f.readline()
    s = []
    si = site()
    days = [0,31,28,31,30,31,30,31,31,30,31,30,31]
    for i in f:
        i = i.strip().split(',')
        id,y,m,r = i[0],int(i[1]),int(i[2]),i[3]
        if y%400==0 or (y%100!=0 and y%4==0):
            days[2] = 29
        r = float(r)
        r = r * 10000.0/3600.0/24.0/float(days[m])
        for j in sites:
            if j.id == id:
                si = j
                break
        if year==0 or (year+last<=y and year-last>=y) :
            if r > 9900000:
                print 'error data:',id,y,m,r
                continue
            if si == site():
                print 'unkown site',id
                continue
            s.append([id,si.lat,si.lon,si.alt,y,m,r])
    return s
'''
sites = []
for i in range(10):
    s = site(i,str(i),i,i,i,i,str(i),i)
    sites.append(s)
for i in range(10):
    sites[i].show()
sites[0].name = '56146'
sites[0].data =sites[0].data + ['xxxxx']
sites[0].data =sites[0].data + ['sdsdds']
for i in range(10):
    print sites[i].show()
'''
