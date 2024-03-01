import pygame
import random
import numpy as np
pygame.init()

# Define the screen dimensions
screen_width = 800
screen_height = 600

# Create the game window
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("RoboCatcher")

# Game loop flag
running = True

# Define game elements
robot_width = 15
robot_height = 15
robot_color = (255, 0, 0)  # Red color
robot_speed = 5
robot_rect = pygame.Rect(50, 50, robot_width, robot_height)  # Red rectangle for the robot
target_position = (400, 300)  # Green circle for the target
target_radius = 20
obstacle_color = (139, 69, 19)  # Brown color
obstacle_width = 30
obstacle_height = 30
num_obstacles = 50

# Discretize the state space
num_state_bins_x = 40
num_state_bins_y = 30
state_width_bin = screen_width // num_state_bins_x
state_height_bin = screen_height // num_state_bins_y

# Q-Learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
num_episodes = 1000

# Initialize Q-table with zeros
num_actions = 4  # Four possible actions: up, down, left, right
q_table = np.zeros((num_state_bins_x, num_state_bins_y, num_actions))

# Generate random obstacle positions that do not overlap with the robot or target
obstacle_positions = []

def check_collision(rect1, rect2):
    return rect1.colliderect(rect2)

while len(obstacle_positions) < num_obstacles:
    obstacle_x = random.randint(0, screen_width - obstacle_width)
    obstacle_y = random.randint(0, screen_height - obstacle_height)
    obstacle_rect = pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height)

    # Check for collisions with the robot and the target
    if not (check_collision(obstacle_rect, robot_rect) or
            check_collision(obstacle_rect, pygame.Rect(*target_position, target_radius * 2, target_radius * 2))):
        obstacle_positions.append((obstacle_x, obstacle_y))

def get_state():
    # Discretize the robot's position to match the Q-table dimensions
    x_bin = min(int(robot_rect.centerx) // state_width_bin, num_state_bins_x - 1)
    y_bin = min(int(robot_rect.centery) // state_height_bin, num_state_bins_y - 1)
    return (x_bin, y_bin)

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # Choose a random action with epsilon probability for exploration
        return random.randint(0, num_actions - 1)
    else:
        # Choose the action with the highest Q-value for exploitation
        return np.argmax(q_table[state[0], state[1]])

def take_action(action):
    if action == 0:  # Move up
        robot_rect.move_ip(0, -robot_speed)
    elif action == 1:  # Move down
        robot_rect.move_ip(0, robot_speed)
    elif action == 2:  # Move left
        robot_rect.move_ip(-robot_speed, 0)
    elif action == 3:  # Move right
        robot_rect.move_ip(robot_speed, 0)

def get_reward():
    if robot_rect.colliderect(pygame.Rect(*target_position, target_radius * 2, target_radius * 2)):
        return 1  # Reward for reaching the target
    elif any(robot_rect.colliderect(pygame.Rect(*obstacle_pos, obstacle_width, obstacle_height)) for obstacle_pos in obstacle_positions):
        return -1  # Penalty for hitting an obstacle
    else:
        return 0

def reset_game():
    # Reset the robot's position to its initial position
    robot_rect.topleft = (50, 50)

# Training loop
for episode in range(num_episodes):
    state = get_state()
    done = False

    while not done:
        action = choose_action(state)
        take_action(action)
        next_state = get_state()
        reward = get_reward()

        # Q-Learning update
        q_table[state[0], state[1], action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action])

        if reward == -1:
            # Robot hit an obstacle, reset the game
            reset_game()
            done = True
        elif reward == 1:
            done = True
        else:
            state = next_state

# Main game loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw game elements (robot, target, obstacles)
    pygame.draw.rect(screen, robot_color, robot_rect)  # Draw robot (red rectangle)
    pygame.draw.circle(screen, (0, 255, 0), target_position, target_radius)  # Draw target (green circle)

    # Draw obstacles
    for obstacle_pos in obstacle_positions:
        pygame.draw.rect(screen, obstacle_color,
                         pygame.Rect(obstacle_pos[0], obstacle_pos[1], obstacle_width, obstacle_height))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
