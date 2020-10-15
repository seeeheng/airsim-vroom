import airsim
import time
import numpy as np

class AirsimEnv():
	def __init__(self, client):
		self.client = client

	def compute_reward(self):
		collision_info = self.client.get_collision_info()
		reward = -1
		if collision_info.has_collided:
			print("Collided with {}".format(collision_info.object_name))
			if collision_info.object_name == "RLTarget_0":
				reward += 100
			elif collision_info.object_name != "RLTarget_0":
				reward -= -20
		print("Reward = {}".format(reward))
		return reward

	def is_done(self):
		collision_info = self.client.get_collision_info()
		if collision_info.has_collided:
			return True
		else:
			return False

# Sample of collision info
"""
<CollisionInfo> {   'has_collided': True,
    'impact_point': <Vector3r> {   'x_val': 23.136844635009766,
    'y_val': -0.7587664723396301,
    'z_val': -0.4420730471611023},
    'normal': <Vector3r> {   'x_val': -1.0,
    'y_val': 0.0,
    'z_val': -0.0},
    'object_id': 241,
    'object_name': 'TemplateCube_Rounded_84',
    'penetration_depth': 0.03684478625655174,
    'position': <Vector3r> {   'x_val': 21.154447555541992,
    'y_val': 0.006373672280460596,
    'z_val': 0.2329985797405243},
    'time_stamp': 1602735741096291000}
"""

class AirsimClient():
    def __init__(self):
        self.client = airsim.CarClient()
        self.client_setup()
        self.car_controls = airsim.CarControls()

    def client_setup(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True)

    def interpret_actions(self,action):
        self.car_controls.brake=0
        self.car_controls.throttle=1
        # Action 0 = brake
        if action == 0:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        # Action 1 = center steering wheel and throttle.
        elif action == 1:
            self.car_controls.steering = 0
        # Action 2 = Turn steering wheel right and throttle.
        elif action == 2:
            self.car_controls.steering = 0.5
        # Action 3 = Turn steering wheel left and throttle.
        elif action == 3:
            self.car_controls.steering = -0.5
        # Action 4 = Turn steering wheel slight right and throttle.
        elif action == 4:
            self.car_controls.steering = 0.25
        # Action 5 = Turn steering wheel slight left and throttle.
        elif action == 5:
            self.car_controls.steering = -0.25
        return self.car_controls

    def act(self):
        self.client.setCarControls(self.car_controls)

    def get_car_state(self):
        car_state = self.client.getCarState()
        return car_state

    def get_collision_info(self):
        collision_info = self.client.simGetCollisionInfo()
        return collision_info

    def get_image(self):
        # Converts tate into a 84x84 numpy array
        responses = self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        current_image = self.transform_input(responses)
        return current_image

    def transform_input(self,responses):
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

        from PIL import Image
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert('L')) 

        return im_final