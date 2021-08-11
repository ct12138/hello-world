import PyLidar3
import time # Time module
import numpy as np
#Serial port to which lidar connected, Get it from device manager windows
#In linux type in terminal -- ls /dev/tty* 
port = "/dev/ttyUSB0"
#port = "/dev/ttyUSB0" #linux
Obj = PyLidar3.YdLidarX4(port) #PyLidar3.your_version_of_lidar(port,chunk_size) 
print('right port')
try:
    if(Obj.Connect()):
        #print(Obj.GetDeviceInfo())
        gen = Obj.StartScanning()
        t = time.time() # start time 
        while (time.time() - t) < 10: #scan for 30 seconds
            tmp = next(gen)
            print('results', tmp)
            keys = list(tmp.keys())
            keys.sort()
            #print('keys', keys)
            results = np.array([tmp[key] for key in keys])
            compress_rate = int(1)
            results = results.reshape([int(360 / compress_rate), compress_rate]).mean(axis=1)
            #dict_sorted = sorted(tmp.items(), tmp.keys())
            #results = [value for key, value in dict_sorted]
            #print(type(results))
            #print(results)
            #time.sleep(2)
        Obj.StopScanning()
        Obj.Disconnect()
    else:
        print("Error connecting to device")
except KeyboardInterrupt:
    Obj.Disconnect()
