# author : wgx
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
        self.alt=int(alt.replace('(','').replace(')',''))
        self.time=time
        self.data=data
    def show(self):
        print "%s,%s,%s,%s\n" % (self.id,self.name,self.pro,self.data)
def get_daily_data(year=0,last=0):
    import os
    dpath = os.path.split(os.path.realpath(__file__))[0]
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
            for x in xrange(j,l):
                if sites[x].id == i[0]:
                    j = x
                    sites[x].data.append(tmp)
                    break
    f.close()
    for i in sites:
        if i.data==[]:
            sites.remove(i)
    print "get %d sites data!\n" % len(sites)
    return sites
    #for i in sites:
        #i.show()
