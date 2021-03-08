import pygame
from gym_slitherin.envs.config import NAME, MOVEMENT, WINDOW

def update(gameobjects):
    # handle events
    human_objects = [x for x in gameobjects if x.type==NAME.HUMAN and x.isalive]
    snakes = [x for x in gameobjects if x.name==NAME.SNAKE and x.isalive]

    # handle key presses !
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            
        keys = pygame.key.get_pressed()

        for key in keys:
            if keys[pygame.K_LEFT]:
                for obj in human_objects:
                    if not obj.movement_dir in [MOVEMENT.LEFT, MOVEMENT.RIGHT]:
                        obj.move_left()

            elif keys[pygame.K_RIGHT]:
                for obj in human_objects:
                    if not obj.movement_dir in [MOVEMENT.LEFT, MOVEMENT.RIGHT]:
                        obj.move_right()

            elif keys[pygame.K_UP]:
                for obj in human_objects:
                    if not obj.movement_dir in [MOVEMENT.UP, MOVEMENT.DOWN]:
                        obj.move_up()

            elif keys[pygame.K_DOWN]:
                for obj in human_objects:
                    if not obj.movement_dir in [MOVEMENT.UP, MOVEMENT.DOWN]:
                        obj.move_down()

    new_gameobjects = []
    new_gameobjects += gameobjects

    # update objects
    for obj in gameobjects:
        if obj.isalive and obj.name==NAME.SNAKE:
            new_gameobjects += obj.update()

    new_gameobjects = [x for x in new_gameobjects if x.isalive]

    # handle collision
    for i,obj in enumerate(gameobjects):
        if obj.isalive and obj.name==NAME.SNAKE:
            new_gameobjects += obj.collision(gameobjects[:i] + gameobjects[i+1:])

    new_gameobjects = [x for x in new_gameobjects if x.isalive]

    return new_gameobjects

def drawGrid(w, rows, h, cols, surface, pygame):
    sizeBtwnX = w // rows
    sizeBtwnY = h // cols
    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwnX
        pygame.draw.line(surface, (100,100,100), (x,0),(x,h))

    for l in range(cols):
        y = y + sizeBtwnY
        pygame.draw.line(surface, (100,100,100), (0,y),(w,y))
        
def render(surface, width,height, rows, cols, gameobjects):
    surface.fill((0,0,0))
    for gameobject in gameobjects:
        gameobject.draw(surface)
    drawGrid(width,rows, height, cols, surface, pygame)
    pygame.display.update()
