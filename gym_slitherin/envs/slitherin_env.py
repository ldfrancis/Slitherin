import gym
from gym_slitherin.envs.config import WINDOW, NAME, COLOR,MOVEMENT, DELTA, DEATH_REWARD, ACTION
from gym_slitherin.envs.gameobjects import Snake, Food
from gym_slitherin.envs.utils import drawGrid, render
import random
import numpy as np
import pygame
from PIL import Image
import copy
import matplotlib.pyplot as plt

class SlitherinEnv(gym.Env):

    metadata = {"render.modes":["human"]}

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self._action_set = [ACTION.LEFT_TURN, ACTION.RIGHT_TURN, ACTION.KEEEP_STRAIGHT]
        self._set_observation_space()
        self._set_action_space()

        self.init(n_agents)

        

    def init(self, n_agents):

        # initialize n_agents snakes
        self.agents = []
        self.gameobjects = []
        for i in range(n_agents):
            self._instantiate_agent(COLOR.colors[i])

        # initialize foods 
        for i in range(n_agents*3):
            self._instantiate_food()


    def _instantiate_food(self):
        pos = self.get_pos()
        self.gameobjects += [Food(pos)]

    def _instantiate_agent(self, color):
        pos = self.get_pos()
        agent = Snake(color, pos)
        self.gameobjects += [agent]
        self.agents += [agent]

    def step(self, actions): 
        rews, obs, infos = [],[],[]

        if np.sum([a.isalive for a in self.agents])*1 == 0:
            return None,[(0,0)]*self.n_agents,True,[1]*self.n_agents

        for agent, action in zip(self.agents, actions):
            rew, ob, info = self._update(agent, action)
            rews += [rew] 
            obs += [ob]
            infos += [info]
        return obs, rews, 0, infos

    def reset(self):
        obs = []
        self.init(self.n_agents)
        for agent in self.agents:
            other_game_objects = list(set(self.gameobjects)-set([agent]))
            ob , _= self._obtain_image(agent, other_game_objects, 0)
            obs += [ob]
        return obs

    def render(self, pause, mode="human", close=False):
        size = (WINDOW.WIDTH, WINDOW.HEIGHT)
        screen = pygame.display.set_mode(size)
        screen.fill((0,0,0))
        drawGrid(WINDOW.WIDTH, WINDOW.ROW, WINDOW.HEIGHT, WINDOW.COL, screen, pygame=pygame)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.pause = True
                    self.paused()
                if event.key == pygame.K_q:
                    pygame.quit()
                    quit()
        
        for gameobject in self.gameobjects:
            gameobject.draw(screen, pygame=pygame)

        pygame.display.update()

        if pause:
            self.pause = True
            self.paused()

    def paused(self):
        while self.pause:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.pause = False


    def get_pos(self):
        pos = []
        for g in self.gameobjects:
            if g.name == NAME.SNAKE:
                pos += [p.pos for p in g.body]
            else:
                pos += [g.cube.pos]

        p = (random.randint(0,WINDOW.COL-1), random.randint(0,WINDOW.COL-1))
        while len(pos) <= WINDOW.COL**2 and p in pos:
            p = (random.randint(0,WINDOW.COL-1), random.randint(0,WINDOW.COL-1))

        return p

    def plot(self, ob):
        
        plt.imshow(ob)
        plt.show()

    def _update(self, agent, action):

        if not agent.isalive:
            info = 1
            self.gameobjects = [gameobject for gameobject in self.gameobjects if gameobject.isalive]
            other_game_objects = list(set(self.gameobjects) - set([agent]))
            ob = None#, [0,0,0,0],[0,0,0,0]#np.zeros((DELTA*2+1,DELTA*2+1))#[0,0,0,0]#np.zeros([WINDOW.ROW, WINDOW.COL])
            #self.plot(ob)
            return [0,0], ob, info
        else:
            info = 0

            other_game_objects = list(set(self.gameobjects) - set([agent]))
            reward, ob = [0,0], None

            ac = self._action_set[action]

            if ac == ACTION.LEFT_TURN:# and (agent.movement_dir != MOVEMENT.DOWN and agent.movement_dir != MOVEMENT.UP):
                agent.left_turn()
            elif ac == ACTION.RIGHT_TURN:# and (agent.movement_dir != MOVEMENT.UP and agent.movement_dir != MOVEMENT.DOWN):
                agent.right_turn()
            

            agent.handle_turns()
            
            # collision with self and wall // returns food //
            self.gameobjects += agent.handle_collision()
            if not agent.isalive:
                info = 1
                self.gameobjects = [gameobject for gameobject in self.gameobjects if gameobject.isalive]
                other_game_objects = list(set(self.gameobjects) - set([agent]))
                ob, _ = self._obtain_image(agent, other_game_objects, ac)
                reward = DEATH_REWARD, 0
                
            
                return reward, ob, info


            # collision with other game objects
            head_pos = agent.body[0].pos

            other_game_objects = list(set(self.gameobjects) - set([agent]))

            for gameobject in other_game_objects:
                
                # snake collision
                if gameobject.name == NAME.SNAKE:
                    other_positions = [x.pos for x in gameobject.body]

                    if head_pos in other_positions:
                        # print("SNAKE")
                        agent.die()
                        self.gameobjects += agent.asfood
                        reward[0] = DEATH_REWARD
                        info = 1
                        break

                # food collision
                elif gameobject.name == NAME.FOOD:
                    if head_pos == gameobject.cube.pos:
                        reward[0] = gameobject.score
                        gameobject.die()
                        agent.addCube()
                        if gameobject.type == NAME.FOOD:
                            self._instantiate_food()
                        break

            self.gameobjects = [gameobject for gameobject in self.gameobjects if gameobject.isalive]
            other_game_objects = list(set(self.gameobjects) - set([agent]))
            ob, reward[-1] = self._obtain_image(agent, other_game_objects, ac)

            return reward, ob, info
    
    def _obtain_image(self, agent, other_game_objects, ac):
        
        if not agent.isalive:
            return None, 0

        r = 0
        head_pos = agent.body[0].pos
        right = (head_pos[0]+1, head_pos[1])
        left = (head_pos[0]-1, head_pos[1])
        up = (head_pos[0],head_pos[1]-1)
        down = (head_pos[0],head_pos[1]+1)

        directions = [up, down, left, right]
  
        closestfoodpos = None

        head_p = head_pos
        minfdist = np.inf
        for g in other_game_objects:
            if g.name == NAME.FOOD:
                fdist = (np.sum((np.array(g.cube.pos) - np.array(head_p))**2))
                if fdist < minfdist:
                    minfdist = fdist
                    closestfoodpos = g.cube.pos
            
        if closestfoodpos is not None:
            closestfooddirection = np.array([1*(closestfoodpos[1] < head_p[1]), 
                1*(closestfoodpos[1] > head_p[1]), 
                1*(closestfoodpos[0] < head_p[0]), 
                1*(closestfoodpos[0] > 
                head_p[0])])
        else:
            closestfooddirection = [0,0,0,0]

        fruits_rel_pos = []
        opp_head_pos = np.zeros(2, dtype = np.float32)
        opp_tail_pos = np.zeros(2, dtype = np.float32)
        food_pos = np.zeros(2, dtype = np.float32)
        od = 10000
        fd = 10000
        ob = np.zeros(len(directions))
        for g in other_game_objects:
            if g.name==NAME.SNAKE:
                dis = np.sum((np.array(g.body[0].pos) - np.array(head_pos)))
                if dis < od:
                    od = dis
                    opp_head_pos = np.array(np.array(g.body[0].pos) - np.array(head_pos),dtype=np.float32)
                    opp_tail_pos = np.array(np.array(g.body[-1].pos) - np.array(head_pos),dtype=np.float32)
                for b in g.body:
                    ob = ob + np.array([1*(b.pos==p) for p in directions])
                    
                
            if g.name==NAME.FOOD:
                dis = np.sum((np.array(g.cube.pos) - np.array(head_pos)))
                if dis < fd:
                    fd = dis
                    foodpos = np.array(np.array(g.cube.pos) - np.array(head_pos),dtype=np.float32)
                ob = ob + np.array([2*(g.cube.pos==p) for p in directions])
                    
    
        for b in agent.body:
            ob =  ob + np.array([1*(b.pos==p) for p in directions])   

        # wall hori
        ob = ob + np.array([1*((p[0] < 0) or (p[0] >= WINDOW.COL)) for p in directions])
        
        # wall vert
        ob = ob+np.array([1*(p[1] < 0 or p[1] >= WINDOW.ROW) for p in directions])
        
        obstacles = np.array(ob,dtype=np.float32)

        # relative tail pos
        tpos = np.array(np.array(agent.body[-1].pos) - np.array(head_pos),dtype=np.float32)

        #fruits relative positions
        obst = np.zeros(3, dtype=np.float32)
        if agent.movement_dir == MOVEMENT.UP:
            x = foodpos[0]
            y = foodpos[1]
            x1 = opp_head_pos[0]
            y1 = opp_head_pos[1]

            foodpos[0] = x
            foodpos[1] = -y

            opp_head_pos[0] = y1
            opp_head_pos[1] = -y1

            obst[1] = obstacles[0]
            obst[0] = obstacles[-2]
            obst[2] = obstacles[-1]

        elif agent.movement_dir == MOVEMENT.DOWN:
            x = foodpos[0]
            y = foodpos[1]
            x1 = opp_head_pos[0]
            y1 = opp_head_pos[1]

            foodpos[0] = -x
            foodpos[1] = y

            opp_head_pos[0] = -x1
            opp_head_pos[1] = y1

            obst[0] = obstacles[-1]
            obst[1] = obstacles[1]
            obst[2] = obstacles[-1]


        elif agent.movement_dir == MOVEMENT.LEFT:
            x = foodpos[0]
            y = foodpos[1]
            x1 = opp_head_pos[0]
            y1 = opp_head_pos[1]

            foodpos[0] = -y
            foodpos[1] = -x

            opp_head_pos[0] = -y1
            opp_head_pos[1] = -x1

            obst[0] = obstacles[1]
            obst[1] = obstacles[-2]
            obst[2] = obstacles[0]

        elif agent.movement_dir == MOVEMENT.RIGHT:
            x = foodpos[0]
            y = foodpos[1]
            x1 = opp_head_pos[0]
            y1 = opp_head_pos[1]

            foodpos[0] = y
            foodpos[1] = x

            opp_head_pos[0] = y1
            opp_head_pos[1] = x1

            obst[0] = obstacles[0]
            obst[1] = obstacles[-1]
            obst[2] = obstacles[1]

        # # up
        if ac == ACTION.LEFT_TURN and foodpos[0] < 0:
            r = 1
        elif ac == ACTION.RIGHT_TURN and foodpos[0] > 0:
            r = 1
        elif ac == ACTION.KEEEP_STRAIGHT and foodpos[1] > 0:
            r = 1
        

        return  (obst, foodpos, opp_head_pos), r  

    def _set_observation_space(self):
        self.observation_space = gym.spaces.Tuple(
            [gym.spaces.Box(low=0, high=255, shape=(WINDOW.ROW, WINDOW.COL), dtype=np.uint8) for i in range(self.n_agents)]
        )

    def _set_action_space(self):
        self.action_space = gym.spaces.Tuple(
            [gym.spaces.Discrete(len(self._action_set)) for i in range(self.n_agents)]
        )
