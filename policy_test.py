import gym
import gym_slitherin
from model import *
import random
from pathlib import Path
from utils import *
import argparse

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--env', dest='env',
                        help='number of snake in env',
                        default=2, type=int)
    parser.add_argument('--model', dest='model',
                        help='trained model to use',
                        default=None, type=str)

    args = parser.parse_args()

    return args

args = parse_args()

env_nb_snakes = args.env
if not env_nb_snakes in [2,4]:
    raise Exception("Can only create env with 2 or 4 snakes")

env = gym.make(f"Slitherin{4}-v0")

# set the obs/act dim all agents have the same observation and action space
obs_dim = [9,9,1]
act_dim = env.action_space[0].shape
action_space = env.action_space[0]
act_no = action_space.n

# parameters
steps_per_epoch = 4000 
gamma = 0.99
lam = 0.97
epochs = 500
max_ep_len = 1000
save_freq = 10
nagents = env.n_agents
trackret = [-1000]*nagents


actors, _ = create_actor_critics(nagents, obs_dim, act_no)
 
# Experience buffer
local_steps_per_epoch = int(steps_per_epoch)
buf = [PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, nagents, gamma, lam) for _ in range(nagents)]#[PPOBuffer([obs_dim[0], obs_dim[1], 1], act_dim, local_steps_per_epoch+10, gamma, lam) for _ in range(env.n_agents)]
tbuf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch*nagents, nagents, gamma, lam)#) for _ in range(nagents)]

# print("reset env")
obs, rs, d, ep_ret, ep_len = env.reset(), [(0,0)]*nagents, False, 0, 0
infos = [0]*nagents
acs = [0]*nagents
lobs = [None,None,None,None]

name = args.model
if name==None:
    raise Exception("must specify the model to use")

# checkpoints
def _create_cpts(n, actors):
  managers = []

  for i in range(n):
    actor_optim = tf.keras.optimizers.Adam(5e-3)
    critic_optim = tf.keras.optimizers.Adam(5e-3)

    
    actor_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=actor_optim, net=actors[i])
    actor_manager = tf.train.CheckpointManager(actor_ckpt, f'./models/{name}/actor/actor{i}/tf_ckpts', max_to_keep=3)
    actor_ckpt.restore(actor_manager.latest_checkpoint)


    if actor_manager.latest_checkpoint:
      print("Restored actor from {}".format(actor_manager.latest_checkpoint))
    else:
      print("Initializing actor from scratch.")

    managers += [(actor_manager, actor_optim, actor_ckpt)]

  return managers

_create_cpts(nagents, actors)

pause = False

while True:
    total_reward_sum = 0
    add = 0

    avg_ret_ind = [0]*nagents
    ep_lens = [0]*nagents

    aoptim = random.randint(0, 1)

    buf_reward = [0]*nagents
    curr_t = 0
    
    acs = [0]*nagents
    
    for i in range(nagents):
        if infos[i] == 0:
            o = obs[i]
            ninput = prepare_obs_inf(o)
            a, logp_t, _ = obtain_alogp(actors[i](ninput), act_no)
            acs[i] = a[0]
        else:
            acs[i] = 0

    # take actions and get reward
    obs, rs, d, infos = env.step([act for act in acs]) 
    
    ep_ret += np.sum([r[0] for r in rs])
    ep_len += 1
    
    env.render(pause)
    terminal = d

    # check for terminal states:
    if terminal:
        obs, rs, d, ep_ret, ep_len = env.reset(), [(0,0)]*nagents, False, 0, 0            
        infos = [0]*nagents
        total_reward_sum += ep_ret
        add += 1