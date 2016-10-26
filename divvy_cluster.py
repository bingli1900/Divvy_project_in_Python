# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 02:41:43 2016

@author: Bing
"""

from numpy import *
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import pandas as pd
from mpl_toolkits.basemap import Basemap   
from mpl_toolkits.mplot3d import Axes3D
from random import random

def calc_weekday(item):
    month = int(item[4])
    day = int(item[5])
    ret = 0
    if month==7:
        if ((day-6)%7+1)<6:
            ret = 1
    elif month==8:
        if ((day+31-6)%7+1)<6:
            ret = 1
    return ret
        
def count_on_station(data, maxid, hour):
    influx = [0. for i in range(maxid+1)]
    outflux = [0. for i in range(maxid+1)]    
    for item in data.values:
        if item[3] == hour:
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
    
def station_index_to_id():
    return 0
    
def station_id_to_index():
    return 0
    
def normalize(vecs):
    v = []
    variance = []
    m = []
    for vec in vecs:
        tmp = var(vec)
        tmp2 = mean(vec)
        variance.append(tmp)
        m.append(tmp2)
        if tmp != 0.:
            v.append( (vec-tmp2)/sqrt(tmp))
        else:
            v.append((vec-tmp2))
    return v, array(variance), array(m)

def get_original_from_norm(vec, vares, means):
    origin_vec = vec * sqrt(vares) + means
    return origin_vec
    
path1 = 'C:\\Users\\Bing\\Desktop\\divvy\\Divvy_Trips_2015_07.csv'
df = pd.read_csv(path1, index_col=0, parse_dates=True)
path2 = 'C:\\Users\\Bing\\Desktop\\divvy\\Divvy_Stations_2015.csv'
dfstation = pd.read_csv(path2, index_col=0, parse_dates=True)

data = df.loc[:, ['from_station_id', 'to_station_id']]
starttime = df.loc[:, 'starttime'].values.tolist()

f = lambda x: x.split(' ')[1]
month = lambda x: x.split('/')[0]
day = lambda x: x.split('/')[1]
h = lambda x: x.split
g = lambda x: float(x.split(':')[0])

starttime1 = map(f, starttime)
hour = map(g, starttime1)
monthdata = map(month, starttime)
daydata = map(day, starttime)

data['time'] = starttime1
data['hour'] = hour
data['month'] = monthdata
data['day'] = daydata
print 'data manip done'

data['weekday'] = map(calc_weekday, data.values)
data = data[data.weekday==1]
weekenddata = data[data.weekday==0]
print 'calc done'

station = dfstation.loc[:, ['latitude', 'longitude']]
numofst = station.shape[0]
maxid = station.index[numofst-1]

for time_slice in range(5, 22):
    influx, outflux = count_on_station(data, maxid, time_slice)
    str1 = 'influx_' + str(time_slice)
    str2 = 'outflux_' + str(time_slice)
    station[str1] = add_on_station(station, influx)
    station[str2] = add_on_station(station, outflux)
    print time_slice

for time_slice in range(7, 23):
    influx, outflux = count_on_station(weekenddata, maxid, time_slice)
    str1 = 'weekend_influx_' + str(time_slice)
    str2 = 'weekend_outflux_' + str(time_slice)
    station[str1] = add_on_station(station, influx)
    station[str2] = add_on_station(station, outflux)
    print time_slice
    
original_vecs = station.iloc[:, 2:68].values
locs = len(original_vecs)
features = len(original_vecs[0])
print locs, features
#station num is 474, vecter 34

vecs = original_vecs.T
norm_vecs, vares, means = normalize(vecs)  # shape 34, 474
norm_vecs_t = transpose(norm_vecs)
c = cov(norm_vecs)
w,v = linalg.eig(c)

#construct new features
x = original_vecs.dot(v[:, 0])
y = original_vecs.dot(v[:, 1])
z = original_vecs.dot(v[:, 2])
q = original_vecs.dot(v[:, 3])
u = original_vecs.dot(v[:, 4])

plt.figure(1)
plt.scatter(x, y)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

cluster_num = 8
physical_x = station.values[:, 1]
physical_y = station.values[:, 0]
featurearr = concatenate((x, y, z, q, u, physical_x), axis=0).reshape((6,474)).T
whitened = whiten(featurearr)
centeroids, labels = kmeans2(whitened, cluster_num)
print labels

plt.figure(3)
colors = [(random(),random(),random()) for i in range(10)]
for i in range(cluster_num):
    label = labels==i
    clusterx = physical_x[label]
    clustery = physical_y[label]
    if len(clusterx != 0):
        plt.scatter(clusterx, clustery, c=colors[i], s=25, edgecolor = 'none')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Chicago divvy-bike stations clustering results")
plt.show()