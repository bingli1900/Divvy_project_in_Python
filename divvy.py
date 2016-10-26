# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 02:41:43 2016

@author: Bing
"""

from numpy import *
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten
import pandas as pd
from mpl_toolkits.basemap import Basemap

path1 = 'C:\\Users\\Bing\\Desktop\\divvy\\Divvy_Trips_2015_07.csv'
df = pd.read_csv(path1, index_col=0, parse_dates=True)
path2 = 'C:\\Users\\Bing\\Desktop\\divvy\\Divvy_Stations_2015.csv'
dfstation = pd.read_csv(path2, index_col=0, parse_dates=True)

data = df.loc[:, ['from_station_id', 'to_station_id']]
starttime = df.loc[:, 'starttime'].values.tolist()
f = lambda x: x.split(' ')[1]
g = lambda x: float(x.split(':')[0])
starttime = map(f, starttime)
hour = map(g, starttime)
data['time'] = starttime
data['hour'] = hour
morningdata = data[data.hour>6]
morningdata = morningdata[morningdata.hour<9]
print morningdata.count()

station = dfstation.loc[:, ['latitude', 'longitude']]
numofst = station.shape[0]
maxid = station.index[numofst-1]

plt.figure(1)
#show the districution of the flux of each station
#using size of dots or color to stand for its in/out flow direction
#so first step would be do statistics for each station
def count_on_station(data, maxid):
    influx = [0. for i in range(maxid+1)]
    outflux = [0. for i in range(maxid+1)]    
    for item in data.values:
        idto = item[1]
        idfrom = item[0]
        influx[idto] += 1
        outflux[idfrom] += 1
    return influx, outflux
    
def add_on_station(station, influx):
    ids = station.index
    ret = ids.tolist()
    for i in range(len(ret)):
        ret[i] = influx[ids[i]]
    return ret
        
influx, outflux = count_on_station(morningdata, maxid)
station['influx'] = add_on_station(station, influx)
station['outflux'] = add_on_station(station, outflux)

x = station.values[:, 1]
y = station.values[:, 0]
maxdot = max(influx)
dotsize_in = array(influx)/maxdot*80.0
dotsize_out= array(outflux)/max(outflux)*80.0
plt.scatter(x, y, s=dotsize_in, c='b')

plt.figure(3)
#include basemap, which is the map background
margin = 0.05
bound_x1 = min(x)-margin
bound_x2 = max(x)+margin
bound_y1 = min(y)
bound_y2 = max(y)

m = Basemap(llcrnrlon=bound_x1,llcrnrlat=bound_y1,urcrnrlon=bound_x2,urcrnrlat=bound_y2,
                projection='merc',lat_ts=5,resolution='h')
m.drawcoastlines()
m.fillcontinents(color='none', lake_color='aqua')
m.drawstates()
x, y = m(x, y)
m.scatter(x, y, s=dotsize_in, c='b',alpha=1.0)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Chicago divvy bike flow map\n during morning rush hours")
plt.show()