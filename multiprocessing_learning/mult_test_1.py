import pandas as pd
import numpy as np
import time

###
# with num_test_samples = 100, it takes about 10 seconds to run this code


start_time = time.time()

batch1 = pd.read_parquet('data/batch_1.parquet')
sensor_geom = pd.read_csv('data/sensor_geometry.csv')

# meta = pd.DataFrame(range())

num_sensors = 5160
event_ids = list(set(batch1.index))

# Make a function that outputs (x,y,z) for a sensor_id input
def id_to_xyz(sen):
    row = tuple(sensor_geom.loc[sen][1:4])
    return row

df_train = pd.DataFrame(columns=[str(i) + 'x' for i in range(0, num_sensors)]
                         +[str(i) + 'y' for i in range(0, num_sensors)]
                         +[str(i) + 'z' for i in range(0, num_sensors)]
                         +['time_' + str(i) for i in range(0, num_sensors)]
                         +['az','ze'])

#### multiprocessing
import multiprocessing as mp




####


count = 0
num_test_samples = 100
batch1_no_aux = batch1[batch1.auxiliary==False]

for index in event_ids[:num_test_samples]:    
    event = batch1_no_aux.loc[index]
    event_copy = event.copy()
    
    event.set_index('sensor_id',
                    inplace=True)
    event = event.groupby('sensor_id').sum()
    
    sensors = event.index
    
    times = [event_copy[event_copy.sensor_id==s].time.values[0] for s in sensors]

    for sensor in sensors:
        sensor_coords = id_to_xyz(sensor)
        sensor_charge = event.loc[sensor,'charge']
        df_train.loc[index, str(sensor)+'x'] = sensor_coords[0]*sensor_charge
        df_train.loc[index, str(sensor)+'y'] = sensor_coords[1]*sensor_charge
        df_train.loc[index, str(sensor)+'z'] = sensor_coords[2]*sensor_charge
        
        # Time
        first_time = event_copy[event_copy.sensor_id==sensor].time.values[0]
        df_train.loc[index, 'time_' + str(sensor)] == first_time
    
    az, ze = np.pi, np.pi/2
    df_train.loc[index,'az'] = az
    df_train.loc[index,'ze'] = ze

    count = count + 1
    if count % 10 == 0:
        print("Working on event", count)

df_train.fillna(0,inplace=True)

print(df_train.head())

print()

print('Runtime (seconds):', time.time() - start_time)