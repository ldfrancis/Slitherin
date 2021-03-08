import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, Flatten, Concatenate, LSTM
from tensorflow.keras.models import Model
import numpy as np
from scipy import signal

"""
Uses implementation from spinning up
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, nagents, gamma=0.99, lam=0.95):
        
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size+10
        self.init(obs_dim, act_dim, size, nagents)

    def init(self, obs_dim, act_dim, size, nagents):

     
        self.obs_buf = np.zeros([size]+[3], dtype=np.float32)
        self.food_buf = np.zeros([size]+[2], dtype=np.float32)
        self.ehead_buf = np.zeros([size]+[2], dtype=np.float32)
        
        
        self.act_buf = np.zeros([size]+ list(act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        

    def store(self, obs, act, rew, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        
        self.obs_buf[self.ptr] = obs[0]
        self.food_buf[self.ptr] = obs[1]
        self.ehead_buf[self.ptr] = obs[2]

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        # self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        # self.obstacle_buf[self.ptr] = o
        # self.food_buf[self.ptr] = f
        # self.enemy_buf[self.ptr] = e

        self.ptr += 1
        self.path_slice = None

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        # vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        # deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        path_slice = slice(0, self.path_start_idx)
        # print(self.path_start_idx)

        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = np.mean(self.adv_buf[path_slice]), np.std(self.adv_buf[path_slice])
        ret_mean, ret_std = np.mean(self.ret_buf[path_slice]), np.std(self.ret_buf[path_slice])
        self.ret_buf[path_slice] = (self.ret_buf[path_slice] - ret_mean) / ret_std

        obs = [self.obs_buf[path_slice],
                self.food_buf[path_slice],
                self.ehead_buf[path_slice],
            ]

        return [obs, self.act_buf[path_slice], 
                self.ret_buf[path_slice], self.logp_buf[path_slice]]

def actor_critic(act_no, nm):
    # actor
    obstacles = Input(shape=(3,), name=f"obs_input")
    food = Input(shape=(2,), name="food")
    enemy_head = Input(shape=(2,), name="h_enemy")
    

    mlpinput = Concatenate(name="concat")([obstacles,food,enemy_head])


    x1 = Dense(4, activation="relu", name="dense1")(mlpinput)

    logits = Dense(act_no, name="logits")(x1)
    
    # critic
    x2 = Concatenate(name=f"concat2")([obstacles,food,enemy_head])

    x2 = Dense(4, activation="relu", name="dense2")(x2)

    y2 = Dense(1, name="out")(x1)
    
    
    actor = Model(inputs=[obstacles, food, enemy_head], outputs=[logits])
    critic = Model(inputs=[obstacles, food, enemy_head], outputs=[y2])
    
    return (actor, critic)


def obtain_alogp(logits, act_dim, a=None):
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.random.categorical(logits,1), axis=1)
    if type(a) == type(None):
        logp = None
    else:
        logp = tf.reduce_sum(tf.one_hot(tf.cast(a,tf.int32), depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    
    return pi, logp_pi, logp

def actor_loss(actor, obslist, a, act_dim, logp_old, adv, clip_ratio=0.2):
    # print([ob.shape for ob in obslist])
    logits = actor(obslist)
    _, _, logp = obtain_alogp(logits, act_dim, a)
    ratio = tf.exp(logp - logp_old)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv>0, (1+clip_ratio)*adv, (1-clip_ratio)*adv)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))
    approx_kl = tf.reduce_mean(logp_old - logp)  

    return pi_loss, approx_kl

def critic_loss(critic, obslist,ret):
    v = critic(obslist)
    v_loss = tf.reduce_mean((ret - v)**2)

    return v_loss

def train_actor(actor, obslist, a, act_dim, logp_old, adv, optimizer, clip_ratio=0.2, target_kl = 0.015):
    with tf.GradientTape() as tape:
        pi_loss, kl = actor_loss(actor, obslist,  a, act_dim, logp_old, adv, clip_ratio=0.2)
    grads = tape.gradient(pi_loss, actor.trainable_variables)
    optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    return kl < target_kl, kl, pi_loss.numpy()


def train_critic(critic, obslist, ret, optimizer):
    with tf.GradientTape() as tape:
        v_loss = critic_loss(critic, obslist, ret)
    grads = tape.gradient(v_loss, critic.trainable_variables)
    optimizer.apply_gradients(zip(grads, critic.trainable_variables))

    return True, v_loss.numpy()