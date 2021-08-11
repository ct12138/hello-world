# Minicar 2020.05
# Follower 2 (f2) script

# The 2nd follower trys keeping the same velocity as the leader,
# and dynamically changes its speed according to control of the leader,
# distances to pred and leader (by ultrasound & comm)

# 'Control' here means the set velocity, for our 1st-order system

# In a loop, the follower checks uart, checks udp msg to update information
# and control regularly, also it broadcasts its own status based on certain comm setting

import tensorflow as tf
import time
import socket
import select
import serial
from RL_brain import PolicyGradient
import gc
import numpy as np

gc.enable()

IP_ADDR = '192.168.3.103'
EPISODES = 100000
TEST = 10
STATEMAX = np.array([2000, 0])
pre_distance = 600
action_threshold = 0.5
update_period = 50

# Ideal distance
IDEAL_DISTANCE = 230  # mm  (200 + 30)

# Controller Parameter
K_X_LEAD = 0.1
K_X_PRED = 0.1

# Tuning Parameter
TURN_RIGHT_PARAM = 10
TURN_LEFT_PARAM = 6
PD_P = 2
PD_D = 0
eps = 0.1

# Frequency
COMM_CYCLE = 0.08  # s
CONTROL_CYCLE = 0.005  # s



class minicar_f2:
    def __init__(self, ideal_distance=IDEAL_DISTANCE):
        # Interface initialization
        self.ser = serial.Serial('/dev/ttyTHS1', 115200)
        # self.ser = serial.Serial()
        # if not self.ser.isOpen():
        #     self.ser.open()

        # self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.sock.bind(('', 9999))
        print('Interface initialized!')

        self.ideal_distance = ideal_distance

        # Status initialization
        self.lead_control = 0
        self.l0f1_distance = self.ideal_distance
        self.f1f2_distance = self.ideal_distance

        self.set_velocity = 0
        self.set_angvel = 0

        self.started = 0  # starts when receives the first packet from leader
        self.last_comm_time = time.time()

        # calibrate camera
        # (content, addr) = self.sock.recvfrom(6) # receive 6 bytes
        # while addr[0] != '127.0.0.1': # only wait for detection
        #     (content,addr) = self.sock.recvfrom(6)

        # self.camera_center = (content[4] + content[5])/255.0/2.0
        # print("Camera Calibrated! Center: %f" % self.camera_center)
        # self.last_bbox = [content[2]/255.0, content[3]/255.0, content[4]/255.0, content[5]/255.0]
        # self.bbox = [content[2]/255.0, content[3]/255.0, content[4]/255.0, content[5]/255.0]

        # self.ser_send(is_init=True)
        self.last_control_time = time.time()

        self.queue = 0
        self.state = np.array([0, 0])
        self.action = 0
        self.reward = 0

    def update_control(self):
        '''Update control based on current bbox & distance/control info'''
        
        self.set_velocity = 20

        control = self.action
        
        if self.f1f2_distance <= pre_distance:
            if control < 0:
                self.set_angvel = -int(np.around(2 * 3.14 * control * 4 * TURN_RIGHT_PARAM))
            else:
                self.set_angvel = -int(np.around(2 * 3.14 * control * 4 * TURN_LEFT_PARAM))
        else:
            self.set_angvel = 0

        print('set_angvel', self.set_angvel)

    def ser_send(self, is_init=False):
        '''Control the lower body'''

        # velocity (int): 0~255
        # advance (int): 0 = forward, 1 = stop, 2 = backward
        # angvel (int): 0~90
        # direction (int): 0 = left, 1 = straight, 2 = right

        if self.set_velocity >= 0:
            advance, velocity = 0, self.set_velocity
        elif self.set_velocity < 0:
            advance, velocity = 2, -self.set_velocity

        if self.set_angvel > 0:
            direction, angvel = 0, self.set_angvel
        elif self.set_angvel < 0:
            direction, angvel = 2, -self.set_angvel
        else:
            direction, angvel = 1, self.set_angvel

        if is_init == False:

            content = b'\xff\xfe' + int(velocity).to_bytes(length=1, byteorder='big') + \
                      int(advance).to_bytes(length=1, byteorder='big') + \
                      int(angvel).to_bytes(length=1, byteorder='big') + \
                      int(direction).to_bytes(length=1, byteorder='big') + b'\x00\x00\x00\x00'
        else:  # Init the base
            print('Base Init!')
            content = b'\xff\xfe' + int(0).to_bytes(length=1, byteorder='big') + \
                      int(0).to_bytes(length=1, byteorder='big') + \
                      int(0).to_bytes(length=1, byteorder='big') + \
                      int(0).to_bytes(length=1, byteorder='big') + b'\x01\x00\x00\x00'

        self.ser.write(content)  # 10 bytes

    def ser_recv(self):
        '''Receive ultrasound from lower body'''

        if self.ser.inWaiting() > 0:
            content = self.ser.read(10)
            measure = content[7] * 256 + content[8]
            self.f1f2_distance = content[7] * 256 + content[8]
            self.ser.reset_input_buffer()

    def rl_update(self, RL, epoch):
        this_state = np.array([self.f1f2_distance, self.f1f2_distance - self.state[0]])
        if (self.state != STATEMAX).any():
            RL.store_transition(self.state, self.action, self.reward)
            if epoch % update_period == 0:
                RL.learn()
            self.action = RL.choose_action(this_state)
            self.reward = (self.state[0] < pre_distance) * self.state[0] * (abs(self.action) - action_threshold)
        self.state = this_state
        print('state', self.state)


def main():
    minicar = minicar_f2(IDEAL_DISTANCE)
    epoch = 0
    try:
        RL = PolicyGradient(
            n_actions=2,
            n_features=1,
            learning_rate=0.02,
            reward_decay=0.99,
            # output_graph=True,
        )
        minicar.state = STATEMAX
        minicar.ser_send(is_init=True)
        print(minicar.state)

        print('Start Following!')
        while True:
            print('epoch', epoch)
            epoch += 1
            # 1. check serial
            minicar.ser_recv()
            # 接收距离信息
            # 2. update rl
            minicar.rl_update(RL, epoch)
            # 更新RL网络，作出决策
            # 3. update control
            minicar.update_control()
            # 更新运动控制策略
            # 4. send serial (control lower body) if overtime
            if (time.time() - minicar.last_control_time) > CONTROL_CYCLE:
                minicar.ser_send()
                minicar.last_control_time = time.time()
            # 传输控制命令
            time.sleep(0.001)  # sleep 1 ms

    except KeyboardInterrupt:
        minicar.set_velocity = 0
        minicar.ser_send()
        saver = tf.train.Saver()
        saver.save(RL.sess,"./model/model.ckpt")


main()
