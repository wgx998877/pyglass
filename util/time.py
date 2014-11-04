def num_to_day(str):
    year = str[:4]
    num = int(str[4:])
    mon_num = [31,28,31,30,31,30,31,31,30,31,30,31]
    if is_r(int(year)):
        mon_num[1] = 29
    m = 0
    d = 0
    for i in range(12):
        if num - mon_num[i] > 0 :
            num -= mon_num[i]
        else :
            m = i + 1
            d = num
            break
    return "%4s%02d%02d" % (year,m,d)
def is_r(year):
    r = False
    if year % 4==0:
        r = True
    if year%100==0 and year%400!=0:
        r = False
    return r
