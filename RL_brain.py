"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

pre_distance = 600
action_threshold = 0.5
hidden_layer = 20

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs, self.act_history = [], [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_actions], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, 5], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None], name="actions_value")
        # fc1

        h = tf.layers.dense(
            self.tf_obs,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        self.up_probability = tf.layers.dense(
            h,
            units=6,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(stddev=0.1))

        #self.act_all_prob = tf.nn.sigmoid(self.act_all_tmp / 100, name='act_all_tmp')
        #self.act_all_prob = (self.act_all_prob - 0.5) * 2
        self.act_all_prob = tf.nn.softmax(self.up_probability)

        with tf.name_scope('loss'):
            #loss = tf.reduce_sum(-tf.to_float(self.tf_obs < pre_distance) * self.tf_obs * (abs(self.act_all_prob) - action_threshold))
            #self.loss = tf.losses.log_loss(
            #labels=self.tf_acts,
            #predictions=self.act_all_prob)
            self.loss0 = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_acts, logits=self.up_probability)
            self.loss = -self.loss0 * self.tf_vt
            
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    '''
            self.w1 = tf.Variable(tf.random_normal([self.n_actions, hidden_layer], stddev=1, seed=1))
            self.b1 = tf.Variable(tf.random_normal([hidden_layer,], stddev=1, seed=1))
            self.w2 = tf.Variable(tf.random_normal([hidden_layer, self.n_features], stddev=1, seed=1))
            self.b2 = tf.Variable(tf.random_normal([self.n_features,], stddev=1, seed=1))
            self.a = tf.add(tf.matmul(self.tf_obs, self.w1), self.b1)
            #b=tf.nn.batch_normalization(b, mean=[0 0], variance=[[1 0],[0 1]], offset=None, scale=None, variance_epsilon=0.0001)
            #self.a = tf.nn.batch_normalization(self.a, mean=0, variance=1, offset=None, scale=None, variance_epsilon=0.0001)
            self.act_all_tmp = tf.add(tf.matmul(self.a, self.w2), self.b2)
            #self.act_all_tmp = tf.nn.batch_normalization(self.act_all_tmp, mean=0, variance=1, offset=None, scale=None, variance_epsilon=0.0001)
            #self.act_history.append(self.act_all_tmp)
            #self.act_history = tf.nn.batch_normalization(self.act_history[:,:,None,None], mean=0, variance=1, offset=None, scale=None, variance_epsilon=0.0001)
            #self.act_history = self.act_history + tf.random_normal(tf.shape(self.act_history), mean=0, stddev=tf.reduce_mean(self.act_history) / 10)
    '''

    def choose_action(self, observation):
        #observation = np.array([observation])
        print('middle', self.sess.run(self.up_probability, feed_dict={self.tf_obs: observation[np.newaxis, :]}))
        outputs = self.sess.run(self.act_all_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        print('outputs',outputs)
        action = outputs[0].argmax(0) - 2.5
        print('action', action)
        return action

    '''
            observation = observation / np.mean(observation)
            print('observation',observation)
            prob_weights = self.sess.run(self.act_all_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
            print('prob_weights',prob_weights)
            prob_0 = self.sess.run(self.act_all_tmp, feed_dict={self.tf_obs: observation[np.newaxis, :]})
            print('prob_0', prob_0)
            if prob_weights[0][0] >= 0.7:
                action = 1
            elif prob_weights[0][0] <= 0.3:
                action = 0
            else:
                action = 0.5
            #act = np.array([[[5,101],[2394,2]],[[5,101],[2394,21]]])
            #act = np.ones((2,2,1,1))
            #act[0,0,0,0]=2
            #act = tf.nn.batch_normalization(act, mean=0, variance=1, offset=None, scale=None,
              #                                           variance_epsilon=0.0001)
            #print('act',self.sess.run(act))
            print('action', action)
            return action
    '''

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        #print('reward_size',discounted_ep_rs_norm.shape)
        #print('reward', np.array(self.ep_rs).sum())
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs), # shape=[None, n_obs]
             self.tf_acts: np.vstack(self.ep_as), # shape=[None, n_obs]
             self.tf_vt: discounted_ep_rs_norm  # shape=[None]
        })
        # np.array([np.array(self.ep_rs).sum()])
        var = tf.trainable_variables()
        var_val = self.sess.run(var)
        for var0, val in zip(var, var_val):
            print("var:{}, value:{}".format(var0.name, val))
        #loss = tf.reduce_sum(-tf.to_float(self.tf_obs < pre_distance) * self.tf_obs * (abs(self.act_all_prob) - action_threshold))
        print('loss0',self.sess.run(self.loss0, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),# shape=[None, n_obs]
             self.tf_acts: np.vstack(self.ep_as), # shape=[None, n_obs]
             self.tf_vt: np.array([np.array(self.ep_rs).sum()])  # shape=[None, n_obs]
        }))
        #print('reward',np.array([np.array(self.ep_rs).sum()]))

        print('loss',self.sess.run(self.loss, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),# shape=[None, n_obs]
             self.tf_acts: np.vstack(self.ep_as), # shape=[None, n_obs]
             self.tf_vt: np.array([np.array(self.ep_rs).sum()])  # shape=[None, n_obs]
        }))

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        print('reward', self.ep_rs)
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)
        discounted_ep_rs = discounted_ep_rs / np.std(discounted_ep_rs)
        discounted_ep_rs = np.resize(discounted_ep_rs,1)
        return discounted_ep_rs



