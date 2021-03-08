from model import *

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



def create_actor_critics(nagents, obs_dim, act_no):
  actors = []
  critics = []
  for i in range(nagents):
    actor, critic = actor_critic(act_no, f"{i}")
    print(f"{i} -> actor_params: {actor.count_params()} || critic_params: {critic.count_params()}")
    actors += [actor]
    critics += [critic]
  return actors, critics


def create_cpts(name, n, actors):
  managers = []


  for i in range(n):
    actor_optim = tf.keras.optimizers.Adam(5e-3)
    critic_optim = tf.keras.optimizers.Adam(5e-3)

    
    actor_ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=actor_optim, net=actors[i])
    actor_manager = tf.train.CheckpointManager(actor_ckpt, f'./{name}/actor/actor{i}/tf_ckpts', max_to_keep=3)
    actor_ckpt.restore(actor_manager.latest_checkpoint)


    if actor_manager.latest_checkpoint:
      print("Restored actor from {}".format(actor_manager.latest_checkpoint))
    else:
      print("Initializing actor from scratch.")


    managers += [(actor_manager, actor_optim, actor_ckpt)]

  return managers
  
