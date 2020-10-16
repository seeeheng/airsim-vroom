import numpy as np
import time
import airsim
import torch

from AirsimClient import AirsimClient
from AirsimEnv import AirsimEnv
from RLAgent import Agent

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
client = AirsimClient()
env = AirsimEnv(client)
agent = Agent(84*84*4,6,device)

current_image = client.get_image()
next_state = agent.process_image(current_image)
start_time = time.time()
n_episodes = 0
env.log_episodes_and_time(n_episodes,time.time())
while True:
    state = next_state
    action = agent.act(state)
    client.interpret_actions(action)
    client.act()

    reward = env.compute_reward()
    done = env.is_done()
    current_reward = 0

    if done:
        if n_episodes % 1 == 0:
            env.log_episodes_and_time(n_episodes,time.time())
            torch.save(agent.q_network_online.state_dict(), 'checkpoints/checkpoint_{}.pth'.format(n_episodes))

        next_image = client.get_image()
        next_state = agent.process_image(next_image)
        agent.step(state, action, reward, next_state, done)
        client.reset()
        car_control= client.interpret_actions(0)
        client.act()
        time.sleep(1)
        n_episodes += 1

    next_image = client.get_image()
    next_state = agent.process_image(next_image)
    agent.step(state, action, reward, next_state, done)

    time.sleep(1)

# Sample of current_image
""" 
[[ 7  8  8 ...  0  0  0]                           
 [ 8  8  8 ...  0  0  0]
 [ 8  8  8 ...  0  0  0]                                                                              
 ...             
 [91 92 93 ... 93 92 91]
 [94 95 96 ... 96 95 94]                                                                              
 [96 97 98 ... 98 97 96]] 
"""

# Sample of car_state
"""
{   'gear': 0,                                                                            
    'handbrake': False,                                                                               
    'kinematics_estimated': <KinematicsState> {   'angular_acceleration': <Vector3r> {   'x_val': 0.0,                                                                                                     
    'y_val': 0.0, 
    'z_val': 0.0},                                                                                    
    'angular_velocity': <Vector3r> {   'x_val': 0.0,                                                                                                                                                       
    'y_val': 0.0,  
    'z_val': 0.0},                                 
    'linear_acceleration': <Vector3r> {   'x_val': 0.0,                                                                                                                                                    
    'y_val': 0.0,                                  
    'z_val': 0.0},                                 
    'linear_velocity': <Vector3r> {   'x_val': 0.0,                                                                                                                                                        
    'y_val': 0.0,                                  
    'z_val': -0.0},                                
    'orientation': <Quaternionr> {   'w_val': 1.0,                                                   
    'x_val': 3.14368044200819e-05,                                                                   
    'y_val': 5.9604641222676946e-08,                                                                 
    'z_val': -1.873779343491977e-12},
    'position': <Vector3r> {   'x_val': 0.0,                                                         
    'y_val': 4.577636616431846e-07, 
    'z_val': 0.24152274429798126}},  
    'maxrpm': 7500.0,                              
    'rpm': 0.0,                                                                                       
    'speed': 0.0,                                                                                     
    'timestamp': 1602734345355201000}
"""