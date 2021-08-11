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
from RL_brain2 import PolicyGradient
import gc
import numpy as np

gc.enable()

IP_ADDR = '192.168.3.103'
EPISODES = 100000
TEST = 10
#STATEMAX = np.array([10000])
STATEMAX = 10000
pre_distance = 800
action_threshold = 50 / 120

# Ideal distance
IDEAL_DISTANCE = 230  # mm  (200 + 30)

# Controller Parameter
K_X_LEAD = 0.1
K_X_PRED = 0.1

# Tuning Parameter
TURN_RIGHT_PARAM = 25
TURN_LEFT_PARAM = 9
PD_P = 2
PD_D = 0
eps = 0

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

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('', 10000))
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
        self.state = 0
        self.action = 0
        self.reward = 0

    def update_control(self):
        '''Update control based on current bbox & distance/control info'''
        '''
        self.set_velocity = self.started * (self.lead_control + K_X_LEAD * (self.l0f1_distance + self.f1f2_distance \
                                                                            - self.ideal_distance * 2) + K_X_PRED * (
                                                        self.f1f2_distance - self.ideal_distance))

        # For this experiment, no backwards is allowed
        self.set_velocity = self.set_velocity * (self.set_velocity > 0)

        if self.set_velocity > 120:
            self.set_velocity = 120
        '''
        '''
        last_deviation = (self.last_bbox[2] + self.last_bbox[3]) / 2.0 - self.camera_center
        deviation = (self.bbox[2] + self.bbox[3]) / 2.0 - self.camera_center

        deviation_r = ((deviation > 0.05) * (deviation - 0.05) + (deviation < -0.05) * (deviation + 0.05))
        last_deviation_r = ((last_deviation > 0.05) * (last_deviation - 0.05) + (last_deviation < -0.05) * (
                    last_deviation + 0.05))

        # control = PD_P * deviation + PD_D * (deviation - last_deviation)
        control = PD_P * deviation_r + PD_D * (deviation_r - last_deviation_r)

        if control >= 0:
            self.set_angvel = -int(round(2 * 3.14 * control * TURN_RIGHT_PARAM))
        else:
            self.set_angvel = -int(round(2 * 3.14 * control * TURN_LEFT_PARAM))
        '''
        self.set_velocity = 20
        # control = self.action - np.around(self.action)
        control = self.action
        print('control', control)

        if control >= 1:
            control = 1
        elif control <= -1:
            control = -1
        
        if self.f1f2_distance <= pre_distance:
            if self.set_angvel < 0:
                #self.set_angvel = -int(np.around(2 * 3.14 * control * 2 * TURN_RIGHT_PARAM))
                set_angvel = -int(np.around(2 * 3.14 * (pre_distance - self.f1f2_distance) / pre_distance \
                                                 * TURN_RIGHT_PARAM))

            elif self.set_angvel > 0:
                #self.set_angvel = -int(np.around(2 * 3.14 * control * 2 * TURN_LEFT_PARAM))
                set_angvel = int(np.around(2 * 3.14 * (pre_distance - self.f1f2_distance) / pre_distance \
                                                 * TURN_LEFT_PARAM))
            else:
                #set_tmp = np.random.randint(0, 2)
                set_tmp = 0
                print('set',set_tmp)
                set_angvel = int(np.around(4 * 3.14 * (set_tmp - 0.5) * (pre_distance - self.f1f2_distance) / pre_distance \
                                           * (set_tmp * TURN_LEFT_PARAM + (1 - set_tmp) * TURN_RIGHT_PARAM)))
        else:
            set_angvel = 0
        
        #if abs(set_angvel) > abs(self.set_angvel):
        #    self.set_angvel = set_angvel
        #else:
        #    self.set_angvel = int(set_angvel * eps + self.set_angvel * (1 - eps))
        '''
        if self.f1f2_distance <= pre_distance:
            if control < 0:
                set_angvel = -int(np.around(2 * 3.14 * control * 4 * TURN_RIGHT_PARAM))
            else:
                set_angvel = -int(np.around(2 * 3.14 * control * 4 * TURN_LEFT_PARAM))
        else:
            set_angvel = 0
        '''
        self.set_angvel = set_angvel
        print('set_angvel', self.set_angvel)

        '''
        if self.set_velocity < 20:
            self.set_angvel = 0
        elif self.set_velocity < 50:
            self.set_angvel /= 2
        '''
        '''
        if deviation > 0.1:       # turn right
            self.queue += -1
        elif deviation > 0.2:
            self.queue += -2
        elif deviation > 0.25:
            self.queue += -3
        elif deviation < -0.1:    # turn left
            self.queue += 1
        elif deviation < -0.2:
            self.queue += 2
        elif deviation < -0.25:
            self.queue += 3

        if self.queue >= 6:
            self.queue -= 6
            self.set_angvel = 1
        elif self.queue <= -6:
            self.queue += 6
            self.set_angvel = -1
        else:
            self.set_angvel = 0
        '''

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
            # direction, angvel = 0, 0
        elif self.set_angvel < 0:
            direction, angvel = 2, -self.set_angvel
        else:
            direction, angvel = 1, self.set_angvel
            # direction, angvel = 2, 0

        if is_init == False:

            # print(self.set_angvel, time.time())
            content = b'\xff\xfe' + int(velocity).to_bytes(length=1, byteorder='big') + \
                      int(advance).to_bytes(length=1, byteorder='big') + \
                      int(angvel).to_bytes(length=1, byteorder='big') + \
                      int(direction).to_bytes(length=1, byteorder='big') + b'\x00\x00\x00\x00'
            #print('angvel', int(angvel).to_bytes(length=1, byteorder='big'))
            # print('content',content)
        else:  # Init the base
            print('Base Init!')
            content = b'\xff\xfe' + int(0).to_bytes(length=1, byteorder='big') + \
                      int(0).to_bytes(length=1, byteorder='big') + \
                      int(0).to_bytes(length=1, byteorder='big') + \
                      int(0).to_bytes(length=1, byteorder='big') + b'\x01\x00\x00\x00'
            # print('content',content)

        self.ser.write(content)  # 10 bytes

    def ser_recv(self):
        '''Receive ultrasound from lower body'''

        if self.ser.inWaiting() > 0:
            content = self.ser.read(10)
            measure = content[7] * 256 + content[8]
            # if measure < 500:
            self.f1f2_distance = content[7] * 256 + content[8]
            self.ser.reset_input_buffer()

    def rl_update(self, RL, epoch):
        '''
        action = agent.noise_action(self.state)
        next_state, reward, done, _ = env.step(action)
        agent.perceive(state, action, reward, next_state, done)
        state = next_state
        print('state', state)
        '''
        # this_state = np.array([self.f1f2_distance])
        this_state = self.f1f2_distance
        if self.state != STATEMAX:
            # agent.perceive(self.state, self.action, self.reward, this_state, 1)
            RL.store_transition(self.state, self.action, self.reward)
            if epoch % 10 == 0:
                RL.learn()
            # self.action = agent.noise_action(this_state)
            self.action = RL.choose_action(this_state)
            self.reward = (self.state < pre_distance) * self.state * (abs(self.action) - action_threshold)
            # self.state = this_state
            # print('state',self.state)
            #print('action', self.action)
            #print('reward', self.reward)
        # print('this_state',self.state)
        self.state = this_state
        print('state', self.state)

    # def udp_send(self):
    #     ''' Send information from this car to others, this car need not'''
    #     pass
    #
    # def udp_recv(self):
    #     '''Receive either car-velocity(192.168.1.10X) or detection-bbox(127.0.0.1)'''
    #
    #     # Use select to check if there is a UDP msg
    #     ready_to_read, ready_to_write, in_error = select.select([self.sock], [], [], 0)
    #     if self.sock in ready_to_read:
    #         (content, addr) = self.sock.recvfrom(6)  # receive 6 bytes
    #
    #         if addr[0] == '127.0.0.1':  # from detection
    #             for i in range(4):
    #                 self.last_bbox[i] = self.bbox[i]
    #                 self.bbox[i] = content[i + 2] / 255.0
    #
    #         elif addr[0] == '192.168.3.101':  # from leader
    #             # print("Recv from leader!")
    #             self.started = 1  # starts to update control
    #             if content[4] == 255:  # emergency stop
    #                 self.started = 0
    #             self.lead_control = content[0] * ((content[1] == 0) - (content[1] == 2))
    #
    #         elif addr[0] == '192.168.3.102':  # from follower 1
    #             # print("Recv from follower 1!")
    #             self.l0f1_distance = content[4] * 256 + content[5]


def main():
    minicar = minicar_f2(IDEAL_DISTANCE)
    epoch = 0
    try:
        # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
        # env = traffic_junction_env.TrafficJunctionEnv()
        # agent = DDPG(env)
        # agent = DDPG()
        RL = PolicyGradient(
            n_actions=1,
            n_features=1,
            learning_rate=0.01,
            reward_decay=0.99,
            # output_graph=True,
        )
        # env.monitor.start('experiments/' + ENV_NAME, force=True)
        # state = env.reset()
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
            # minicar.rl_update(agent)
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
