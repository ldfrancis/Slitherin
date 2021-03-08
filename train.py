import gym
import gym_slitherin
from model import *
import os
import random
from pathlib import Path
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
                      help='name to give saved model chkpoints',
                      default="model", type=str)
  parser.add_argument("--render", default=False, action="store_true" , help="render env")
  
  args = parser.parse_args()
  return args

def _expand_ob(x):
    return np.array(np.expand_dims(np.expand_dims(x,-1),0)/(act_no-1.0), dtype=np.float32)

def _expand_ob_store(x):
    return np.array(np.expand_dims(x,-1)/(act_no-1.0), dtype=np.float32)

def prepare_obs_inf(obs):
  obs, food, ehead = obs

  food = np.clip(food, -1, 1)
  ehead = np.clip(ehead, -1, 1)

  return [np.expand_dims(obs, 0)/2, np.expand_dims(food, 0), np.expand_dims(ehead,0)]

def prepare_obs_store(obs):
  obs, food, ehead = obs

  food = np.clip(food, -1, 1)
  ehead = np.clip(ehead, -1, 1)

  return [obs/2, food, ehead]


def create_actor_critics(nagents, act_no):
  actors = []
  critics = [] 
  for i in range(nagents):
    actor, critic = actor_critic(act_no, f"{i}")
    print(f"{i} -> actor_params: {actor.count_params()} || critic_params: {critic.count_params()}")
    actors += [actor]
    critics += [critic]
  return actors, critics

# checkpoints
def create_cpts(n, actors, critics, name):
  managers = []


  for i in range(n):
    actor_optim = tf.keras.optimizers.Adam(5e-3)
    critic_optim = tf.keras.optimizers.Adam(5e-3)

    
    actor_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=actor_optim, net=actors[i])
    actor_manager = tf.train.CheckpointManager(actor_ckpt, f'./models/{name}/actor/actor{i}/tf_ckpts', max_to_keep=3)
    actor_ckpt.restore(actor_manager.latest_checkpoint)

    
    critic_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=critic_optim, net=critics[i])
    critic_manager = tf.train.CheckpointManager(critic_ckpt, f'./models/{name}/critic/critic{i}/tf_ckpts', max_to_keep=3)
    critic_ckpt.restore(critic_manager.latest_checkpoint)

    if actor_manager.latest_checkpoint:
      print("Restored actor from {}".format(actor_manager.latest_checkpoint))
    else:
      print("Initializing actor from scratch.")

    if critic_manager.latest_checkpoint:
      print("Restored critic from {}".format(critic_manager.latest_checkpoint))
    else:
      print("Initializing critic from scratch.")

    managers += [(actor_manager, actor_optim, actor_ckpt, critic_manager, critic_optim, critic_ckpt)]

  return managers
  

def update(actor, critic, manager, a, c, buf, act_no):
    actor_manager, actor_optim, actor_ckpt, critic_manager, critic_optim, critic_ckpt = manager
    results = buf.get()

    if True:
        save_path = actor_manager.save()
    
    for i in range(a):
        res = results
        ta, kl, loss = train_actor(actor, res[0], res[1], act_no, res[3], res[2], actor_optim, target_kl=0.09)
        actor_ckpt.step.assign_add(1)
        
        if not ta:
            print(f"large KL at actor training {i} KL:{kl}")
            break

def main():

  args = parse_args()

  env_nb_snakes = args.env
  render = args.render
  model = args.model
  if not env_nb_snakes in [2,4]:
    raise Exception("Can only create env with 2 or 4 snakes")

  env = gym.make(f"Slitherin{env_nb_snakes}-v0")

  os.makedirs("models", exist_ok=True)

  # set the obs/act dim all agents have the same observation and action space
  obs_dim = [9,9,1]
  act_dim = env.action_space[0].shape
  action_space = env.action_space[0]
  act_no = action_space.n
  print(act_no)

  # parameters
  steps_per_epoch = 4000 
  gamma = 0.99
  lam = 0.97
  epochs = 500
  max_ep_len = 1000
  save_freq = 10
  nagents = env.n_agents
  trackret = [-1000]*nagents


  actors, critics = create_actor_critics(nagents, act_no)
 
  # Experience buffer
  local_steps_per_epoch = int(steps_per_epoch)
  buf = [PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, nagents, gamma, lam) for _ in range(nagents)]#[PPOBuffer([obs_dim[0], obs_dim[1], 1], act_dim, local_steps_per_epoch+10, gamma, lam) for _ in range(env.n_agents)]
  tbuf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch*nagents, nagents, gamma, lam)#) for _ in range(nagents)]

  # reset environment
  obs, rs, d, ep_ret, ep_len = env.reset(), [(0,0)]*nagents, False, 0, 0
  infos = [0]*nagents
  acs = [0]*nagents
  lobs = [None,None,None,None]

  # base path
  base = Path("SLITHERIN’/")

  managers = create_cpts(nagents, actors, critics, model)

  # load in or create training stats (used to resume training and keep track of
  # performance)
  with open("stats.txt", "r+") as f:
    contents = f.read()
  vals = contents.split("\n")
  if len(vals) > 0:
    try:
      l_epoch = int(vals[-2].split(" ")[0])+1
    except:
      l_epoch = 0

  epoch = 0
  pause = False

  # Main loop: collect experience in env and update/log each epoch
  for ep in range(l_epoch, 4000):
      total_reward_sum = 0
      add = 0

      avg_ret_ind = [0]*nagents
      ep_lens = [0]*nagents

      aoptim = random.randint(0, 1)

      buf_reward = [0]*nagents
      curr_t = 0
      for t in range(local_steps_per_epoch):
          acs = [0]*nagents
          fobs = obs
          finfos = infos
          v_ts = [0]*nagents
          logp_ts = [0]*nagents
          for i in range(nagents):
            if infos[i] == 0:
              o = obs[i]
              ninput = prepare_obs_inf(o)
              a, logp_t, _ = obtain_alogp(actors[i](ninput), act_no)
              logp_ts[i] = logp_t
              acs[i] = a[0]
            else:
              acs[i] = 0

          # take actions and get reward
          obs, rs, d, infos = env.step([act for act in acs]) 
          pause = False
          for i in range(nagents):
            reward = rs[i]
            env_reward = reward[0]*30
            curr_reward = 0
            
            if finfos[i] == 0:
              o = fobs[i]
              buf_reward[i] = curr_reward+env_reward
              buf[i].store(prepare_obs_store(o), acs[i], buf_reward[i], logp_ts[i])
              
          ep_ret += np.sum([r[0] for r in rs])
          ep_len += 1
          
          if render:
            env.render(pause)
          terminal = d or (ep_len == max_ep_len)

          # check for terminal states:
          curr_t += 1
          if terminal or (t==local_steps_per_epoch-1):
              curr_t = 0
              if not(terminal):
                  print('Warning: trajectory cut off by epoch at %d steps. t:%d, local_steps_per_epoch:%d'%(ep_len, t, local_steps_per_epoch))
                  

              # if trajectory didn't reach terminal state, bootstrap value target
              for i in range(nagents):
                last_val = buf_reward[i] if infos[i] else -100#critics[i]([prepare_obs_inf(obs[i])])[0,0].numpy()
                buf[i].finish_path(last_val)
              if terminal:
                  # only save EpRet / EpLen if trajectory finished
                  total_reward_sum += ep_ret
                  add += 1
              obs, rs, d, ep_ret, ep_len = env.reset(), [(0,0)]*nagents, False, 0, 0
              
              infos = [0]*nagents
      
      for i in range(nagents):
        update(actors[i], critics[i], managers[i], 10, 10, buf[i],act_no)
      
      print(f"epoch: {ep} || average_ret: {total_reward_sum/add}")
      with open("stats.txt", "a+") as f:
        f.write(f"{ep} {total_reward_sum/add}\n")

if __name__ == "__main__":
  main()