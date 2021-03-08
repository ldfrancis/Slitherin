from gym_slitherin.envs.config import WINDOW, NAME, MOVEMENT
import random

# CUBE
class Cube(object):
    rows = WINDOW.ROW
    cols = WINDOW.COL
    w = WINDOW.WIDTH
    h = WINDOW.HEIGHT
    def __init__(self,start,dirnx=1,dirny=0,color=(0,255,0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color
 
        
    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def set_pos(self, pos):
        self.pos = pos

    def draw(self, surface, eyes=False, pygame=None):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]
        pygame.draw.rect(surface, self.color, (i*dis+1,j*dis+1, dis-2, dis-2))

        if eyes:
            centre = dis//2
            radius = 3
            circleMiddle = (i*dis+centre,j*dis+centre)
            pygame.draw.circle(surface, (0,0,0), circleMiddle, radius)
      

# FOOD
class Food(object):
    name = NAME.FOOD
    def __init__(self,pos, score=1, objtype=NAME.FOOD):
        self.cube = Cube(pos)
        self.score = 1
        self.isalive = True
        self.type = objtype
    
    def draw(self, surface, pygame):
        if self.isalive:
            self.cube.draw(surface, pygame=pygame)

    def die(self):
        self.isalive = False


# SNAKE
class Snake(object):

    name = NAME.SNAKE

    def __init__(self, color, pos, objtype="AI"):
        self.turns = {}
        self.isalive = True
        self.movement_dir = MOVEMENT.RIGHT
        self.score = 0

        self.color = color
        self.body = []
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.type = objtype

        for _ in range(7):
            self.addCube()

    def set_color(self,color):
        for body in self.body:
            body.color = color

    def left_turn(self):
        if self.movement_dir == MOVEMENT.UP:
            self.dirnx = -1
            self.dirny = 0
            self.movement_dir = MOVEMENT.LEFT
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif self.movement_dir == MOVEMENT.DOWN:
            self.dirnx = +1
            self.dirny = 0
            self.movement_dir = MOVEMENT.RIGHT
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif self.movement_dir == MOVEMENT.RIGHT:
            self.dirnx = 0
            self.dirny = -1
            self.movement_dir = MOVEMENT.UP
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif self.movement_dir == MOVEMENT.LEFT:
            self.dirnx = 0
            self.dirny = +1
            self.movement_dir = MOVEMENT.DOWN
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

    def right_turn(self):
        if self.movement_dir == MOVEMENT.UP:
            self.dirnx = +1
            self.dirny = 0
            self.movement_dir = MOVEMENT.RIGHT
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif self.movement_dir == MOVEMENT.DOWN:
            self.dirnx = -1
            self.dirny = 0
            self.movement_dir = MOVEMENT.LEFT
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif self.movement_dir == MOVEMENT.RIGHT:
            self.dirnx = 0
            self.dirny = +1
            self.movement_dir = MOVEMENT.DOWN
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif self.movement_dir == MOVEMENT.LEFT:
            self.dirnx = 0
            self.dirny = -1
            self.movement_dir = MOVEMENT.UP
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]


    def handle_turns(self):
        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0],turn[1])
                if i == len(self.body)-1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx,c.dirny)

    def update(self):
        # handle collision with self
        foods = self.handle_collision()
        self.handle_turns()
        print(self.body[0].pos)

        return foods

    def handle_collision(self):
        ## collision with self
        head_pos = self.body[0].pos
        body_positions = [x.pos for x in self.body[1:]]
        if head_pos in body_positions:
            self.die()
            return self.asfood
        

        ## collision with wall
        x = head_pos[0]
        y = head_pos[1]
        marg = 1

        if x < 0:
            self.body[0].pos = (0, y)
            self.die()
            return self.asfood

        elif x > WINDOW.ROW - marg:
            self.body[0].pos = (WINDOW.ROW - marg, y)
            self.die()
            return self.asfood
        
        if y < 0:
            self.body[0].pos = (x, 0)
            self.die()
            return self.asfood

        elif y > WINDOW.COL - marg:
            #print(y)
            self.body[0].pos = (x, WINDOW.COL -marg)
            self.die()
            return self.asfood

        return []

    ## collision with other game objects
    def collision(self, gameobjects):
        head_pos = self.body[0].pos
        new_gameobjects = []
       
        for gameobject in gameobjects:
            
            # snake collision
            if gameobject.name == NAME.SNAKE:
                other_positions = [x.pos for x in gameobject.body]

                if head_pos in other_positions:
                    print("SNAKE")
                    self.die()
                    new_gameobjects += self.asfood

            # food collision
            if gameobject.name == NAME.FOOD:
                if head_pos == gameobject.cube.pos:
                    print("FOOD")
                    self.score += gameobject.score
                    gameobject.die()
                    self.addCube()
                    if gameobject.type == NAME.FOOD:
                        new_food = Food((random.randint(0, 20), random.randint(0, 20)), 1)
                        new_gameobjects += [new_food]
        
        return new_gameobjects


    def die(self):
        if len(self.body) >= 4:
            self.body = self.body[1::2]
            self.asfood = [Food(x.pos, 1, "SNAKE") for x in self.body]
        else:
            self.asfood = [Food(x.pos, 1, "SNAKE") for x in self.body[:1]]

        self.isalive = False
        

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0]-1,tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0]+1,tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0],tail.pos[1]-1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0],tail.pos[1]+1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy
        

    def draw(self, surface, pygame):
        if self.isalive:
            for i, c in enumerate(self.body):
                if i ==0:
                    c.draw(surface, True, pygame=pygame)
                else:
                    c.draw(surface, pygame=pygame)

