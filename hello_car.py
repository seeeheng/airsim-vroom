import airsim
import time

# connect to the AirSim simulator
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

def interpret_actions(action):
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

while True:
    # get state of the car
    car_state = client.getCarState()
    print("Speed %d, Gear %d" % (car_state.speed, car_state.gear))
    action_to_take = input("What next? 1-5: ")
    # reset = input("Reset?: ")

    interpret_actions(int(action_to_take))

    # if int(reset):
    #     client.reset()

    # set the controls for car
    # car_controls.throttle = 1
    # car_controls.steering = 1
    client.setCarControls(car_controls)
    # let car drive a bit
    time.sleep(1)

    # # get camera images from the car
    # responses = client.simGetImages([
    #     airsim.ImageRequest(0, airsim.ImageType.DepthVis),
    #     airsim.ImageRequest(1, airsim.ImageType.DepthPlanner, True)])
    # print('Retrieved images: %d', len(responses))

    # # do something with images
    # for response in responses:
    #     if response.pixels_as_float:
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
    #         airsim.write_pfm('py1.pfm', airsim.get_pfm_array(response))
    #     else:
    #         print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
    #         airsim.write_file('py1.png', response.image_data_uint8)