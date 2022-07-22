"""This script loads the data from an experiment and plots it. Support plotting of multiple expriments."""

import scipy.io
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt


plt.close('all')
path = str((Path(__file__).parent.absolute()).parent.absolute()) + "/logs"

# ff_4: Kp = 200, Kd = 18
# ff_5: Kp = 120, Kd = 12
# ff_6: Kp = 200, Kd = 12
# ff_7: Kp = 120, Kd = 18
# ff_8: Kp = 120, Kd = 25
# ff_8: Kp = 100, Kd = 25


data1 = scipy.io.loadmat(os.path.join(path, "exp_springs_ff_8.mat"))
data2 = scipy.io.loadmat(os.path.join(path, "exp_springs_ff_9.mat"))


def plot_normal_forces(data):
    plt.figure()
    for i in range(4):
        plt.plot(data["time"].reshape(-1),data["feet_normal_forces"][:,i],label="Foot {}".format(i))
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Foot Normal Force [N]")
    plt.title("Normal Foot Force vs Time")
    plt.grid()
    plt.xlim(right=8)

    
def plot_normal_forces_comparison(data1,data2):
    for i in range(4):
        plt.figure()
    
        plt.plot(data1["time"].reshape(-1),data1["feet_normal_forces"][:,i],label="No Compensation")
        plt.plot(data2["time"].reshape(-1),data2["feet_normal_forces"][:,i],label="With Compensation")

        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Foot {} Normal Force [N]".format(i))
        plt.title("Normal Force Foot {} vs Time".format(i))
        plt.grid()
        plt.xlim(right=8)


def plot_body_height_comparison(data1,data2):
    plt.figure()

    plt.plot(data1["time"].reshape(-1),data1["base_pos"][:,2],label="No Compensation")
    plt.plot(data2["time"].reshape(-1),data2["base_pos"][:,2],label="With Compensation")

    plt.legend()

    plt.xlabel("Time [s]")
    plt.ylabel("Body height [m]")
    plt.title("Body height vs Time with and without compensation")
    plt.xlim(right=8)
    plt.grid()

def plot_body_ori_comparison(data1,data2):
    

    x = ["roll","pitch","yaw"]
    print(data1["base_ori"].shape)
    for i in range(3):
        plt.figure()
        plt.plot(data1["time"].reshape(-1),data1["base_ori"][:,i],label="No Compensation")
        plt.plot(data2["time"].reshape(-1),data2["base_ori"][:,i],label="With Compensation")

        plt.legend()

        plt.xlabel("Time [s]")
        plt.ylabel("Body orientation {}".format(x[i]))
        plt.title("Body orientation {} vs Time with and without compensation".format(x[i]))
        plt.xlim(right=8)
        plt.grid()


def plot_body_pitch_comparison(data1,data2):
    plt.figure()

    plt.plot(data1["time"].reshape(-1),data1["pitch_rate"][:],label="No Compensation")
    plt.plot(data2["time"].reshape(-1),data2["pitch_rate"][:],label="With Compensation")

    plt.legend()

    plt.xlabel("Time [s]")
    plt.ylabel("Pitch rate")
    plt.title("Pitch rate vs Time with and without compensation")
    plt.grid()
    plt.xlim(right=8)


def plot_foot_pos_comparison(data1,data2):
    dim = ["x","y","z"]
    
    for i in range(3):
        plt.figure()
        
        plt.plot(data1["time"].reshape(-1),data1["foot_pos"][:,0,i],label="No Compensation")
        plt.plot(data2["time"].reshape(-1),data2["foot_pos"][:,0,i],label="With Compensation")

        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Foot 1 position in {} coordinate [m]".format(dim[i]))
        plt.title("Evolution of Foot 1 position in {} coordinate [m]".format(dim[i]))
        plt.grid()
        plt.xlim(right=8)




# plot_normal_forces(data1)
# plot_normal_forces_comparison(data1,data2)
# plot_body_height_comparison(data1,data2)
# plot_foot_pos_comparison(data1,data2)
# plot_body_pitch_comparison(data1,data2)
plot_body_ori_comparison(data1,data2)

plt.show()