import numpy as np
import time
import airsim
from AirsimClient import AirsimClient

while True:
    client = AirsimClient()
    current_state = client.get_state()
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