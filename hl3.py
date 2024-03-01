import pygame
import sys
import random
import numpy as np 

pygame.init()

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BASKET_WIDTH, BASKET_HEIGHT = 80, 60
OBJECT_WIDTH, OBJECT_HEIGHT = 40, 40


bgImage = pygame.image.load(r"D:\HL3\models\blackmesa.jpg")

# HL logo
icon_image = pygame.image.load(r"D:\HL3\models\alyx-removebg-preview.png")
icon_image = pygame.transform.scale(icon_image, (32,32))
pygame.display.set_icon(icon_image)


# falling Scientists
combine = pygame.image.load(r"D:\HL3\models\scientist-removebg-preview.png")  
combine = pygame.transform.scale(combine, (80, 80))  

# ALYX NOT GORDON (NPC)
gordon = pygame.image.load(r"D:\HL3\models\alyx-removebg-preview.png")
gordon = pygame.transform.scale(gordon, (100,50))

# GORDON not alyx
alyx = pygame.image.load(r"D:\HL3\models\gordonfreeman-removebg-preview.png") 
alyx = pygame.transform.scale(alyx, (150,50))



screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("HL3?")



basket_x = (WIDTH - BASKET_WIDTH) // 2
basket_y = HEIGHT - BASKET_HEIGHT

alyx_x = (WIDTH - BASKET_WIDTH) // 2
alyx_y = HEIGHT - BASKET_HEIGHT

object_x = random.randint(0, WIDTH - OBJECT_WIDTH)
object_y = 0
object_speed = 1

clock = pygame.time.Clock()
frameRateCap = 30
# grid cells in the x and y directions
num_x_cells = 10
num_y_cells = 10

# Calculating width and height of each grid cell
grid_cell_width = WIDTH // num_x_cells
grid_cell_height = HEIGHT // num_y_cells

# Q-table
num_actions=2

num_states = (num_x_cells, num_y_cells, num_x_cells, num_y_cells)  # (basket_x, basket_y, object_x, object_y)

Q_table = np.zeros((num_states + (num_actions,)))

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
max_episodes = 1000  # Maximum number of episodes

left_key_pressed = False
right_key_pressed = False 

object_speed = 1
speed_increase_interval = 50000 # 300 frames (10 seconds at 30 FPS)
frame_counter = 0

def move_alyx(direction):
    global alyx_x
    if direction == "left":
        alyx_x -= 1
    elif direction == "right":
        alyx_x += 1
    
    alyx_x = max(0,min(alyx_x, WIDTH - BASKET_WIDTH))


score = 0
for episode in range(max_episodes):
    # Initialize the environment for a new episode
    basket_x = (WIDTH - BASKET_WIDTH) // 2
    basket_y = HEIGHT - BASKET_HEIGHT
    object_x = random.randint(0, WIDTH - OBJECT_WIDTH)
    object_y = 0

    running= True

    counted_combines = []

    
    while running:
        screen.fill((0,0,0))
        screen.blit(bgImage, (0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    left_key_pressed = True
                elif event.key == pygame.K_RIGHT:
                    right_key_pressed = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    left_key_pressed = False
                elif event.key == pygame.K_RIGHT:
                    right_key_pressed = False
        # Calculate the state based on the current positions
        state = (
            min(max(basket_x // grid_cell_width, 0), num_x_cells - 1),
            min(max(basket_y // grid_cell_height, 0), num_y_cells - 1),
            min(max(object_x // grid_cell_width, 0), num_x_cells - 1),
            min(max(object_y // grid_cell_height, 0), num_y_cells - 1)
        )
        
        # Increase object speed every 10 seconds
        frame_counter += 1
        if frame_counter >= speed_increase_interval:
            object_speed += 1
            frame_counter = 0

        
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions - 1)  # Explore
        else:
            action = np.argmax(Q_table[state])  # Exploit

        # Move the basket based on the selected action
        if action == 0:  # Move left
            basket_x -= 5
        else:  # Move right
            basket_x += 5

        # Ensure the basket stays within the screen
        basket_x = max(0, min(basket_x, WIDTH - BASKET_WIDTH))
        
        # Updating alyx pos
        if left_key_pressed:
            move_alyx("left")
        if right_key_pressed:
            move_alyx("right")


        # Update the position of the falling object
        object_y += object_speed

        # Calculate the next state
        next_state_index = (
            min(max(basket_x // grid_cell_width, 0), num_x_cells - 1),
            min(max(basket_y // grid_cell_height, 0), num_y_cells - 1),
            min(max(object_x // grid_cell_width, 0), num_x_cells - 1),
            min(max(object_y // grid_cell_height, 0), num_y_cells - 1)
        )

        # Calculate the reward
        reward = 3 if basket_x < object_x + OBJECT_WIDTH and basket_x + BASKET_WIDTH > object_x else -1

        # Calculate the reward(alyx)

        alyx_reward = 3 if alyx_x < object_x + OBJECT_WIDTH and alyx_x + BASKET_WIDTH > object_x else -1

        if alyx_reward == 3 and object_x not in counted_combines:
            counted_combines.append(object_x)
            score+=1

        if reward == 3 and object_x not in counted_combines:
            counted_combines.append(object_x)
            score+=1
        

        # Update the Q-value for the current state-action pair
        Q_table[state][action] += alpha * (reward + gamma * np.max(Q_table[next_state_index]) - Q_table[state][action])

        
        # Check if the object has reached the bottom
        if object_y > HEIGHT:
            running= False

        #combine
        screen.blit(combine, (object_x, object_y))

        # Display the game as it's being played (optional)
        
        screen.blit(gordon, (basket_x, basket_y))
        
        screen.blit(alyx, (alyx_x, alyx_y))
        font = pygame.font.Font(None, 36)
        text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(text, (10, 10))
        #pygame.draw.rect(screen, BLUE, (basket_x, basket_y, BASKET_WIDTH, BASKET_HEIGHT))
        #pygame.draw.rect(screen, BLUE, (object_x, object_y, OBJECT_WIDTH, OBJECT_HEIGHT))
        pygame.display.update()

        
    # Limit the frame rate
    clock.tick(frameRateCap)



# Quit Pygame
pygame.quit()
sys.exit()
