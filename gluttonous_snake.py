import pygame, time
from pygame.locals import QUIT
from pygame.locals import KEYDOWN, K_RIGHT, K_LEFT, K_UP, K_DOWN, K_ESCAPE

from game import Game
from search_ai import get_move as search_move


black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)

green = pygame.Color(0, 200, 0)
bright_green = pygame.Color(0, 255, 0)
red = pygame.Color(200, 0, 0)
bright_red = pygame.Color(255, 0, 0)
blue = pygame.Color(32, 178, 170)
bright_blue = pygame.Color(32, 200, 200)
yellow =  pygame.Color(255, 205, 0) 
bright_yellow =  pygame.Color(255, 255, 0) 

game = Game()
rect_len = game.settings.rect_len
snake = game.snake
pygame.init()    
fpsClock = pygame.time.Clock()
screen = pygame.display.set_mode((game.settings.width*15, game.settings.height*15))
pygame.display.set_caption('Gluttonous')

crash_sound = pygame.mixer.Sound('./sound/crash.wav') 

def text_objects(text, font, color = black):
    text_surface = font.render(text, True, color)
    return text_surface, text_surface.get_rect()

def message_display(text, x, y, color = black):
    large_text = pygame.font.SysFont('comicsansms', 50)
    text_surf, text_rect = text_objects(text, large_text, color)
    text_rect.center = (x, y)
    screen.blit(text_surf, text_rect)
    pygame.display.update()
    
def button(msg, x, y, w, h, inactive_color, active_color, action = None, parameter = None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x + w > mouse[0] > x and y + h > mouse[1] > y:
        pygame.draw.rect(screen, active_color, (x, y, w, h))
        if click[0] == 1 and action != None:
            if parameter != None:
                action(parameter)    
            else:
                action() 
    else:
        pygame.draw.rect(screen, inactive_color, (x, y, w, h))
        
    smallText = pygame.font.SysFont('comicsansms', 20)
    TextSurf, TextRect = text_objects(msg, smallText)
    TextRect.center = (x + (w / 2), y + (h / 2))
    screen.blit(TextSurf, TextRect)

def quitgame():
    pygame.quit()
    quit()
    
def crash():    
    pygame.mixer.Sound.play(crash_sound)    
    message_display('crashed', game.settings.width/2*15, game.settings.height/3*15, white)  
    time.sleep(1)
   
def initial_interface():
    intro = True
    while intro:
                
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                
        screen.fill(white)
        message_display('Gluttonous', game.settings.width/2*15, game.settings.height/4*15)

        button('Go!', 80, 210, 80, 40, green, bright_green, game_loop, 'human')
        button('Quit', 270, 210, 80, 40, red, bright_red, quitgame)
        button('AI 1', 80, 280, 80, 40, blue, bright_blue, game_loop, 'search_ai')
        button('AI 2', 270, 280, 80, 40, yellow, bright_yellow, DQN)
        
        pygame.display.update()
        pygame.time.Clock().tick(15)
        
def game_loop(player, fps = 30):
    game.restart_game()
    
    while not game.game_end():
                     
        pygame.event.pump()
        
        if player == 'search_ai':
            move = search_move(game) 
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
        else:
            move = human_move()
            fps = 10
                   
        game.do_move(move)
        
        screen.fill(black)
        
        game.snake.blit(rect_len, screen)
        game.strawberry.blit(screen)
        game.blit_score(white, screen)
        
        pygame.display.flip()
        
        fpsClock.tick(fps)  
        
    crash()
    
def human_move():

    direction = snake.facing

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            
        elif event.type == KEYDOWN:
            if event.key == K_RIGHT or event.key == ord('d'):
                direction = 'right'
            if event.key == K_LEFT or event.key == ord('a'):
                direction = 'left'
            if event.key == K_UP or event.key == ord('w'):
                direction = 'up'
            if event.key == K_DOWN or event.key == ord('s'):
                direction = 'down'
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))
    
    move = game.direction_to_int(direction)
    return move

def DQN():
    import tensorflow as tf
    from DQN import DeepQNetwork
    import numpy as np
    
    game.restart_game()
    
    tf.reset_default_graph()
    sess = tf.Session()            

    dqn = DeepQNetwork(sess, game)  
        
    game_state = game.current_state()

    start_state = np.concatenate((game_state, game_state, game_state, game_state), axis=2)
    s_t = start_state
    
    while not game.game_end():
        # choose an action epsilon greedily
        _, action_index = dqn.choose_action(s_t)  
        
        move = action_index
        game.do_move(move)
        
        pygame.event.pump()
        
        game_state = game.current_state()
        s_t = np.append(game_state, s_t[:, :, :-2], axis=2)
        
        screen.fill(black)
        
        game.snake.blit(rect_len, screen)
        game.strawberry.blit(screen)
        game.blit_score(white, screen)
        
        pygame.display.flip()
        
        fpsClock.tick(15)  

    crash()        
                  
if __name__ == "__main__":
    initial_interface()
