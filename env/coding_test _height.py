"""This file will be used to test my code changes"""

import os, inspect
# so we can import files
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
os.sys.path.insert(0, currentdir)

import time, datetime
import numpy as np

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data
import random
random.seed(10)

import quadruped
import configs_a1 as robot_config
import matplotlib.pyplot as plt
import PIL
import imageio

# print(pybullet.isNumpyEnabled())

#################################
# create folder for fallen test
#################################
directory = 'FALLEN_TEST'
path = os.path.abspath(os.path.join(currentdir,
                                    os.pardir,
                                    directory))
if not os.path.exists(path):
    os.makedirs(path)

##################################
# Image parameters
##################################
img_width = 333
img_height = 480
img_renderer = pybullet.ER_TINY_RENDERER

####################################
# Some auxiliary method
####################################
def plot_history(time, state, ylabel):
    fig, axs = plt.subplots(nrows=3, sharex=True)
    titles = ['HIP', 'THIGH', 'CALF']
    labels = ('FR', 'FL', 'RR', 'RL')
    fig.suptitle(ylabel)
    for i, (ax, title) in enumerate(zip(axs, titles)):
        data = state[i + np.array([0, 3, 6, 9]), :]
        ax.plot(time, np.transpose(data))
        ax.set_title(title)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabel)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    # plt.show()
    return fig, axs

def plot_angles(time, state, ylabel):
    def_angles = state[0:3, 0]
    fig, axs = plt.subplots(nrows=3, sharex=True)
    titles = ['HIP', 'THIGH', 'CALF']
    labels = ('FR', 'FL', 'RR', 'RL', 'init_angle')
    fig.suptitle(ylabel)
    for i, (ax, title, def_angle) in enumerate(
                                    zip(axs, titles, def_angles)):
        def_angle = np.full(np.shape(time)[0], def_angle)
        def_angle = np.degrees(def_angle)
        data = state[i + np.array([0, 3, 6, 9]), :]
        data = np.degrees(data)
        ax.plot(time, np.transpose(data))
        ax.plot(time, def_angle)
        ax.set_title(title)
        ax.set_xlabel('t')
        ax.set_ylabel(ylabel)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0,0,0.75,1])
    # plt.show()
    return fig, axs

def plot_height(time, h):
    safe = np.full(np.shape(time)[0], 0.18)
    fig, ax = plt.subplots()
    ax.plot(time, h)
    ax.plot(time, safe, '--')
    ax.legend(('height', 'is fallen'), loc='best')
    ax.set_xlabel('t')
    ax.set_ylabel('h')
    fig.suptitle('h(t)')
    # plt.show()
    return fig, ax

def nicePrint(vec):
    """ Print single vector (list, tuple, or numpy array) """
    # check if vec is a numpy array
    if isinstance(vec,np.ndarray):
        np.set_printoptions(precision=3)
        print(vec)
        return
    currStr = '['
    for x in vec:
        currStr = currStr + '{: .3g} '.format(x)
    currStr += ']'
    return currStr

def restart_simulation(_pybullet_client, robot, height=0):
    _pybullet_client.resetSimulation()
    _pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
    _pybullet_client.setGravity(0,0,-9.81)
    _pybullet_client.loadURDF("plane.urdf")
    robot.Reset_height(h=height)

def change_trunk_mass(_pybullet_client, robot, beta=0):
    """ Modify the trunk mass as new_mass = (1 + beta) * old_mass.
    Obs:
    -It will automatically change the inertia
    -it will not modify the URDF files so in the robot class
        the mass is still setted to the old value

    Args:
        _pybullet_client (bullet_client): bullet_client
        robot (class Quadruped): The class
        beta (int, optional): parameter to change mass value. Defaults to 0.
    """
    trunkId = robot._chassis_link_ids[0]
    old_mass = _pybullet_client.getDynamicsInfo(robot.quadruped,
                                                trunkId)[0]
    new_mass = old_mass * (1 + beta)
    _pybullet_client.changeDynamics(robot.quadruped,
                                    trunkId,
                                    mass=new_mass)

def constraint_z_axis(_pybullet_client, robot):
    _pybullet_client.createConstraint(robot.quadruped,
                                      -1,
                                      -1,
                                      -1,
                                      _pybullet_client.JOINT_PRISMATIC,
                                      [0,0,1],
                                      [0,0,0],
                                      [0,0,0],
                                      childFrameOrientation=[0,0,0,1])

#######################################
# simulation
#######################################
_pybullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
robot = quadruped.Quadruped(pybullet_client=_pybullet_client,
                            robot_config=robot_config,
                            accurate_motor_model_enabled=True,
                            on_rack=False)
motor = robot._motor_model


#########################################
# time and container for values to log
#########################################
time_sleep = 1./2400.
timesteps = 2000
time_sequence = [time_sleep * i for i in range(timesteps)]
history_angles = np.zeros((robot_config.NUM_MOTORS,timesteps))
history_tau = np.zeros((robot_config.NUM_MOTORS,timesteps))
history_tau_spring = np.zeros((robot_config.NUM_MOTORS,timesteps))
history_h = np.zeros(shape=(timesteps,))
img_buffer = np.zeros(shape=(img_height, img_width, 4, timesteps))

###############################
# Define parameters for test
###############################
list_stiffness = [ [15, 8.61, 5], [15, 8.61, 8.34],
                  [15, 10, 10], [15, 15, 15], [15, 16, 8]]
list_rest_angles = [ [0, np.pi/4, -0.71], [0, np.pi/4, -1.05],
                    [0, np.pi/4, -1], [0, np.pi/4, -0.7], [0, np.pi/4, -np.pi/2]]
# _beta = [0, 0, 0, 0]
# _height = [0, 0, 0, 0]
idx_sim = 2

###################################
# Initialize simulation
###################################
# restart_simulation(_pybullet_client, robot=robot, height=_height[idx_sim])
# motor._setSpringStiffness(list_stiffness[idx_sim])
# motor._setSpringRestAngle(list_rest_angles[idx_sim])
# change_trunk_mass(_pybullet_client, robot, beta=_beta[idx_sim])
# constraint_z_axis(_pybullet_client, robot)

default_config = robot_config.INIT_MOTOR_ANGLES

h_fin = 0.8
h_test = np.linspace(0, h_fin, num=10)
h_min = 0.18
success = True
h_max = 0

# for h in h_test:
#     # Initialize height
#     restart_simulation(_pybullet_client, robot=robot, height=h)
#     motor._setSpringStiffness(list_stiffness[idx_sim])
#     motor._setSpringRestAngle(list_rest_angles[idx_sim])
#     constraint_z_axis(_pybullet_client, robot)

#     if success:
#         for i in range (timesteps):
#             robot.ApplySpringAction()
#             # robot.ApplyAction(default_config, enable_springs=True)

#             # Log values
#             angles = robot.GetMotorAngles()
#             taus = robot._applied_motor_torque
#             taus_springs = robot._spring_torque
#             height = robot.getHeight()

#             if height <= h_min:
#                 success = False
            
#             # history_angles[:,i] = angles
#             # history_tau[:,i] = taus
#             # history_tau_spring[:,i] = taus_springs
#             # history_h[i] = height

#             # simulate    
#             _pybullet_client.stepSimulation()
#             time.sleep(time_sleep)
#         if success:
#             h_max = h

# restart_simulation(_pybullet_client, robot=robot, height=h_max)
# motor._setSpringStiffness(list_stiffness[idx_sim])
# motor._setSpringRestAngle(list_rest_angles[idx_sim])
# constraint_z_axis(_pybullet_client, robot)

# for i in range (timesteps):
#     robot.ApplySpringAction()
#     # robot.ApplyAction(default_config, enable_springs=True)

#     # Log values
#     angles = robot.GetMotorAngles()
#     taus = robot._applied_motor_torque
#     taus_springs = robot._spring_torque
#     height = robot.getHeight()
    
#     history_angles[:,i] = angles
#     history_tau[:,i] = taus
#     history_tau_spring[:,i] = taus_springs
#     history_h[i] = height

#     # if i % 800 == 0:
#     #     # Store image
#     #     img=_pybullet_client.getCameraImage(width=img_width,
#     #                                         height=img_height,
#     #                                         renderer=img_renderer)[2]
#     #     # im = PIL.Image.fromarray(img, mode="RGBA")
#     #     img_buffer[:,:,:,i] = img
#     #     print('okay')

#     # simulate    
#     _pybullet_client.stepSimulation()
#     time.sleep(time_sleep)
        
_pybullet_client.disconnect()

print('sim ended')

##########################
# Plot Stuff #
##########################
fig_angles, _ = plot_angles(time_sequence, history_angles, 'angles')
fig_tau, _ = plot_history(time_sequence, history_tau, 'tau')
fig_tau_spring, _ = plot_history(time_sequence, history_tau_spring, 'tau_spring')
fig_h, _ = plot_height(time_sequence, history_h)
figs = [fig_angles, fig_tau_spring, fig_h]
names = ['state', 'tau_spring', 'height']
dict_figs = dict(zip(figs, names))

###########################
# Create sub-folder
###########################
sub_dir_name = f"K={nicePrint(list_stiffness[idx_sim])}&"
sub_dir_name += f"theta_0={nicePrint(list_rest_angles[idx_sim])}&"
sub_dir_name += f"h={h_max:.3g}"
path_fig = os.path.abspath(os.path.join(path, sub_dir_name))
if not os.path.exists(path_fig):
    os.makedirs(path_fig)

###########################
# store data
###########################
for fig, name in dict_figs.items():
    fig.savefig(os.path.join(path_fig, name))
path_txt = os.path.abspath(os.path.join(path_fig, "spring_initial_condition.txt"))
with open(path_txt, "w") as file:
    file.write(f'K_springs = {nicePrint(list_stiffness[idx_sim])}\n')
    file.write(f'theta_rest_springs = {nicePrint(list_rest_angles[idx_sim])}\n')
    file.write(f'initial_height = {h_max:.3g}\n')

###########################
# create video
# ###########################
# Store data
# directory = 'TEMPORARY_IMAGES'
# path_img = os.path.abspath(os.path.join(currentdir,
#                                     os.pardir,
#                                     directory))
# if not os.path.exists(path_img):
#     os.makedirs(path_img)
# for i, array in enumerate(img_buffer):
#     im = PIL.Image.fromarray(array, mode='RGBA')
#     im.save(os.path.abspath(os.path.join(path_img, f"frame_{i}.png")))

import cv2

print('video making')
path_video = os.path.abspath(os.path.join(path_fig, 'video.avi'))

out = cv2.VideoWriter(path_video,
                      cv2.VideoWriter_fourcc(*'DIVX'),
                      240,
                      (img_width,img_height))
for i in range(np.shape(img_buffer)[-1]):
    array = np.copy(img_buffer[:,:,:,i])
    pil_image = (PIL.Image.fromarray(array, mode='RGBA')).convert('RGB')
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    out.write(open_cv_image)
out.release()



print('done')
