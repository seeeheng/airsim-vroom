import torch
import torch.nn.functional as F
import numpy as np
import random
from QNetwork import QNetwork
from ReplayBuffer import ReplayBuffer
from History import History

LEARNING_RATE = 5e-4
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4
NUMBER_BUFFER_FRAMES = 4
SIZEROWS = 84
SIZECOLS = 84

class Agent():
	def __init__(self, state_size, action_size, device, train=True):
		self.state_size = state_size
		self.action_size = action_size
		self.device = device

		self.q_network_online = QNetwork(state_size, action_size).to(device)
		self.q_network_target = QNetwork(state_size, action_size).to(device)
		self.optimizer = torch.optim.Adam(self.q_network_online.parameters(), lr = LEARNING_RATE)

		self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device)
		self.current_step = 0
		self.history = History((NUMBER_BUFFER_FRAMES, SIZEROWS, SIZECOLS))

	def step(self, state, action, reward, next_state, done):
		self.memory.remember(state, action, reward, next_state, done)

		# at every $UPDATE_EVERY steps, sample.
		self.current_step = (self.current_step + 1) % UPDATE_EVERY
		if self.current_step == 0:
			if len(self.memory) > BATCH_SIZE:
				experiences = self.memory.retrieve()
				self.learn(experiences, GAMMA) 

	def learn(self, experiences, GAMMA=0.99):
		states, actions, rewards, next_states, dones = experiences

		self.optimizer.zero_grad()

		qtargets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1) #Bunch of shit here
		QTarget = rewards + (gamma * qtargets_next * (1-dones))
		QExpected = self.q_network_online(states).gather(1, actions)

		loss = F.mse_loss(QExpected, QTarget)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.soft_update(self.q_network_online, self.q_network_target, TAU)

	def process_image(self, image):
		self.history.append(image)
		return self.history.value()

	def act(self, state, eps=0.999):
		state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

		self.q_network_online.eval()
		with torch.no_grad():
			action_values = self.q_network_online(state)
		self.q_network_online.train()

		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size)) # if discrete action space

	def soft_update(self, online_model, target_model, tau):
		for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
			target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data) 

	def save(self):
		torch.save(self.q_network_online, "car_online.pth")
		torch.save(self.q_network_target, "car_target.pth")
		