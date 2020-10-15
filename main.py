import numpy as np
import time
import airsim
from AirsimClient import AirsimClient
from AirsimEnv import AirsimEnv

client = AirsimClient()
env = AirsimEnv(client)

while True:
    current_image = client.get_image()
    # print(current_image)
    
    # action = int(input("Manual drive!: "))
    client.interpret_actions(1)
    car_state = client.get_car_state()
    client.act()

    print("Reward gotten: {}".format(env.compute_reward()))
    print("is_done: {}".format(env.is_done()))

    # print(car_state)
    collision_info = client.get_collision_info()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
    print(collision_info)
    # print(client.car_controls)
    time.sleep(1)
    # get state of the car
    # action = agent.act(current_state)
    # car_controls = interpret_action(action)
    # client.setCarControls(carControls)
    # car_state = client.getCarState()
    # reward = compute_reward(car_state)
    # done = isDone(car_state, car_controls, reward)
    # if done == 1:
        # reward = -10
    # agent.observe(current_state, action, reward, done)
    # agent.train()
    # if done:
        # client.reset()
        # car_control = interpret_action(1)
        # client.setCarControls(car_control)
        # time.sleep(1)
        # current_step += 1


    # responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
    # current_state = transform_input(responses)


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