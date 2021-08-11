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
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1
        self.w1 = tf.Variable(tf.random_normal([1,5], stddev=1, seed=1))
        self.b1 = tf.Variable(tf.random_normal([5,], stddev=1, seed=1))
        self.w2 = tf.Variable(tf.random_normal([5,1], stddev=1, seed=1))
        self.b2 = tf.Variable(tf.random_normal([1,], stddev=1, seed=1))
        self.a = tf.add(tf.matmul(self.tf_obs, self.w1), self.b1)
        self.act_all_tmp = tf.add(tf.matmul(self.a, self.w2), self.b2)
        self.act_history.append(self.act_all_tmp)
        self.act_history = self.act_history + tf.random_normal(tf.shape(self.act_history), mean=0, stddev=tf.reduce_mean(self.act_history) / 10)
        #self.act_all_prob = tf.nn.sigmoid(self.act_all_tmp / tf.reduce_mean(self.act_history), name='act_all_tmp')
        
        self.act_all_prob = tf.nn.sigmoid(self.act_all_tmp / 20000, name='act_all_tmp')
        self.act_all_prob = (self.act_all_prob - 0.5) * 2
        ''' 
        self.layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        print(self.layer)
        print(type(self.layer))
        # fc2
        self.all_act = tf.layers.dense(
            inputs=self.layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(self.all_act, name='act_prob')  # use softmax to convert to probability
        '''

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            #neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #loss = tf.reduce_sum((self.tf_obs < pre_distance) * self.tf_obs * (abs(self.all_act_prob) - action_threshold))
            #loss = tf.reduce_sum(self.tf_obs * (abs(self.act_all_tmp) - action_threshold))
        #loss = tf.square(tf.matmul(tf.matmul(self.tf_obs, self.w1), self.w2) - self.tf_obs)
            loss = tf.reduce_sum(-tf.to_float(self.tf_obs < pre_distance) * self.tf_obs * (abs(self.act_all_prob) - action_threshold))
            #neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        
            #loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            #loss = tf.reduce_mean(self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        #print('obss',observation)
        observation = np.array([observation])
        prob_weights = self.sess.run(self.act_all_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        prob_0 = self.sess.run(self.act_all_tmp, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        print('prob_weights',prob_weights)
        print('prob_0',prob_0)
        #print('all_act_prob',self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        #print('weights',prob_weights.shape[1])
        #print('weightn',prob_weights.ravel())
        #action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        action = prob_weights[0][0]
        #action = np.random.randint(0, 2)
        print('action(state)',action)
        return action

    def store_transition(self, s, a, r):
        #print('obs1',self.ep_obs)
        self.ep_obs.append(s)
        #print('obs2',self.ep_obs)
        #print('s',s)
        self.ep_as.append(a)
        #print('a',a)
        self.ep_rs.append(r)
        #print('r',r)

    def learn(self):
        # discount and normalize episode reward
        # discounted_ep_rs_norm = self._discount_and_norm_rewards()
        discounted_ep_rs_norm = np.resize(self.ep_rs,len(self.ep_rs))
        #print('self.action', np.array(self.ep_as).shape)
        #print(self.ep_obs)
        #print(self.ep_as)
        #print(discounted_ep_rs_norm)
        # train on episode
        self.sess.run(self.train_op, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs)  # shape=[None, n_obs]
             #self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             #self.tf_vt: np.array(discounted_ep_rs_norm),  # shape=[None, ]
        })
        var = tf.trainable_variables()
        var_val = self.sess.run(var)
        for var0, val in zip(var, var_val):
            print("var:{}, value:{}".format(var0.name, val))
        '''
        print('all',self.sess.run(tf.reduce_sum(-tf.log(self.all_act_prob)), feed_dict={self.tf_obs: np.vstack(self.ep_obs)}))
        print('hot',self.sess.run(tf.one_hot(self.tf_acts, self.n_actions), feed_dict={self.tf_acts: np.array(self.ep_as)}))
        neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
        '''
        #loss = tf.reduce_sum(self.tf_obs * (abs(self.all_act_prob) - action_threshold))  # reward guided loss
        #loss = tf.reduce_sum(self.tf_obs * (abs(self.act_all_tmp) - action_threshold))
        #loss = tf.reduce_sum(-self.tf_obs * abs(tf.nn.sigmoid(self.act_all_tmp) - action_threshold))
        loss = tf.reduce_sum(
            -tf.to_float(self.tf_obs < pre_distance) * self.tf_obs * (abs(self.act_all_prob) - action_threshold))
        print('loss',self.sess.run(loss, feed_dict={
             self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
             #self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
             #self.tf_vt: np.array(discounted_ep_rs_norm),  # shape=[None, ]
        }))
        #print('neg_log_prob',neg_log_prob)
        #print('reward',self.tf_vt)
        #print('self.all_act_prob',self.all_act_prob)

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        '''
        print('dis',discounted_ep_rs)
        print(type(discounted_ep_rs))
        print('mean_dis',np.mean(discounted_ep_rs))
        print(type(np.mean(discounted_ep_rs)))
        print(type(float(np.mean(discounted_ep_rs))))
       
        print('formdis',discounted_ep_rs)
        print(type(discounted_ep_rs))
        '''
        discounted_ep_rs = discounted_ep_rs - np.mean(discounted_ep_rs)
        if discounted_ep_rs != 0:
            discounted_ep_rs = discounted_ep_rs / np.std(discounted_ep_rs)
        discounted_ep_rs = np.resize(discounted_ep_rs,1)
        #print('latterdis',discounted_ep_rs)
        #print(discounted_ep_rs.shape)
        return discounted_ep_rs



