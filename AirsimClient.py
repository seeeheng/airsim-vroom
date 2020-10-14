import airsim
import time
import numpy as np

class AirsimClient():
    def __init__(self):
        self.client = airsim.CarClient()

    def client_setup(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()


    def interpret_actions(self,action):
        car_controls.brake=0
        car_controls.throttle=1
        # Action 0 = brake
        if action == 0:
            car_controls.throttle = 0
            car_controls.brake = 1
        # Action 1 = center steering wheel and throttle.
        elif action == 1:
            car_controls.steering = 0
        # Action 2 = Turn steering wheel right and throttle.
        elif action == 2:
            car_controls.steering = 0.5
        # Action 3 = Turn steering wheel left and throttle.
        elif action == 3:
            car_controls.steering = -0.5
        # Action 4 = Turn steering wheel slight right and throttle.
        elif action == 4:
            car_controls.steering = 0.25
        # Action 5 = Turn steering wheel slight left and throttle.
        elif action == 5:
            car_controls.steering = -0.25
        return car_controls

    def get_state(self):
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        current_state = self.transform_input(responses)
        print(current_state)
        return current_state

    def transform_input(self,responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 

        return im_final