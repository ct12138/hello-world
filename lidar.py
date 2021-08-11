import PyLidar3
import time # Time module
import numpy as np
#Serial port to which lidar connected, Get it from device manager windows
#In linux type in terminal -- ls /dev/tty* 
port_default = "/dev/ttyUSB0"
#port = "/dev/ttyUSB0" #linux
class Lidar:
    def __init__(self, port = port_default):
        self.port = port
        self.Obj = PyLidar3.YdLidarX4(self.port) #PyLidar3.your_version_of_lidar(port,chunk_size)
        print('ok')

    def start(self):
        if(self.Obj.Connect()):
            #print(Obj.GetDeviceInfo())
            self.gen = self.Obj.StartScanning()
        else:
            print("Error connecting to device")

    def scan(self):
        tmp = next(self.gen)
        #print('results', tmp)
        keys = list(tmp.keys())
        keys.sort()
        #print('keys', keys)
        results = np.array([tmp[key] for key in keys])
        compress_rate = int(9)
        results = results.reshape([int(360 / compress_rate), compress_rate]).mean(axis=1)
        #dict_sorted = sorted(tmp.items(), tmp.keys())
        #results = [value for key, value in dict_sorted]
        #time.sleep(2)
        return results

    def stop(self):
        self.Obj.StopScanning()
        self.Obj.Disconnect()


   
