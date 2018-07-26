# -*- coding: utf-8 -*-
"""
Created on Sat May  5 18:32:22 2018

@author: zou
"""

from game import Game
import numpy as np
import pygame
import copy

game = Game()
settings = game.settings

#up down left right
direction = [[0,-1], [0,1], [-1,0], [1,0]]

blackColour = pygame.Color(0,0,0)

def is_move_possible(pos, snake):

    if pos in snake.segments[:-1]:
        return False    
    if pos[0] >= settings.width or pos[0] < 0:
        return False
    if pos[1] >= settings.height or pos[1] < 0:
        return False
        
    return True

class Qitem():
    def __init__(self, pos, direction, step, parent=None):
        self.pos = pos
        self.direction = direction
        self.step = step        
        self.parent = parent

    def __repr__(self):
        return str(self.pos)   

def BFS(snake, destination):
    snake_head = snake.segments[0]
    
    mark = np.zeros((settings.width, settings.height))
    finded = False
    
    front = 0
    item = Qitem(snake_head, direction=-1, step=0, parent=-1)
    track = [item]
    
    while len(track) != front: 
        pos = track[front].pos
        step = track[front].step
        
        if pos == destination:
            finded = True
            return finded, track, track[front]           
                          
        for i in range(4):
            move_direction = direction[i]
            new_pos = [pos[i] + move_direction[i] for i in range(2)]
                
            if is_move_possible(new_pos, snake) and mark[tuple(new_pos)] == 0:
                track.append(Qitem(new_pos, i, step+1, parent=front))
                mark[tuple(new_pos)] = 1          
     
        front += 1
    return finded, track, track[-1]

def distance(x, y):
    d = [abs(x[i]-y[i]) for i in range(len(x))]
    return sum(d)

def DFS(snake, destination, pos, mark, track):                
    
    if pos == destination:
        return True
    
    if len(track) == 0:
        distance_order = []
        for i in range(4):
            move_direction = direction[i]
            new_pos = [pos[i] + move_direction[i] for i in range(2)]        
            distance_order.append([distance(new_pos, destination), i])
        
        distance_order = sorted(distance_order, key=lambda x:x[0], reverse=True)
    else:
        distance_order = [(i,i) for i in range(4)]
    
    for _, i in distance_order:
        move_direction = direction[i]
        new_pos = [pos[i] + move_direction[i] for i in range(2)]
           
        if is_move_possible(new_pos, snake) and mark[tuple(new_pos)] == 0:
            track.append([pos, i])
            mark[tuple(new_pos)] = 1  
            if DFS(snake, destination, new_pos, mark, track):
                return True
            track.pop()
    
    return False

def ramdom_choice(snake):
    snake_head = snake.segments[0]
    pos = snake_head
    for i in range(4):
        move_direction = direction[i]
        new_pos = [pos[i] + move_direction[i] for i in range(2)]
            
        if is_move_possible(new_pos, snake):
            return i
        
    return np.random.choice(4)

def tail_accessible(game):   
    
    snake = game.snake
    snake_tail = snake.segments[-1] 

    finded, _ = BFS_move_list(snake, snake_tail)
    return finded

def follow_tail(game):
    snake = game.snake
    snake_tail =  snake.segments[-1]   
    
    finded, move_list = DFS_move_list(snake, snake_tail)
    
    if finded:
        return move_list[0]
    else:
        return ramdom_choice(snake)           
    
def DFS_move_list(snake, destination):
    mark = np.zeros((settings.width, settings.height))

    snake_head = snake.segments[0]
    pos = snake_head
    track = []
    
    finded = DFS(snake, destination, pos, mark, track)
    
    move_list = [move for _,move in track]
    return finded, move_list
    
def BFS_move_list(snake, destination):
    finded, track, last = BFS(snake, destination)
    move_list = [last.direction]
    
    while last.parent != -1:
        last = track[last.parent]
        move_list.append(last.direction)

    move_list = move_list[-2::-1]
    
    return finded, move_list

def find_safe_move(game, move_list):

    virtual_game = Game()
    virtual_game.snake.position = copy.deepcopy(game.snake.position)
    virtual_game.snake.facing = copy.deepcopy(game.snake.facing)
    virtual_game.snake.segments = copy.deepcopy(game.snake.segments)
    virtual_game.strawberry.position = copy.deepcopy(game.strawberry.position)
    
    for move in move_list:
        virtual_game.do_move(move)
        
    if tail_accessible(virtual_game):
        return move_list[0]
    else:
        return follow_tail(game) 
    
def get_move(game):
    snake = game.snake                           
    strawberry = game.strawberry
            
    finded, move_list = BFS_move_list(snake, strawberry.position)
    
    if finded:
        move = find_safe_move(game, move_list)
    else:
        move = follow_tail(game)        
    return move

def search_ai():
    
    game = Game()
    rect_len = game.settings.rect_len
    
    pygame.init()    
    fpsClock = pygame.time.Clock()
    screen = pygame.display.set_mode((game.settings.width*15, game.settings.height*15))
    pygame.display.set_caption('Raspberry Snake')
    
    while not game.game_end():
        pygame.event.pump()
        
        move = get_move(game) 
                   
        game.do_move(move)
        
        screen.fill(blackColour)
        
        game.snake.blit(rect_len, screen)
        game.strawberry.blit(screen)
        game.blit_score(screen)
        
        pygame.display.flip()
        
        fpsClock.tick(30)  

if __name__ == "__main__":
    search_ai()




