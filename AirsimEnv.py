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
			# print("Collided with {}".format(collision_info.object_name))
			if collision_info.object_name == "RLTarget_0":
				reward += 100000
			elif collision_info.object_name != "RLTarget_0":
				reward -= 2000
		# print("Reward = {}".format(reward))
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