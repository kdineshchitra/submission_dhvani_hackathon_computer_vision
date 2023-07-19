# imports
import numpy as np
import plotly.express as px

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Constants & Initial position
# values given in the question
a = 10
b = 28
c = 2.667
x0 = 0
y0 = 1
z0 = 1.05

# Time
t = np.linspace(0, 10, 10)  # Adjust the time range as per the requirements


# Function to calculate derivatives
def dx_dt(x, y, z):
    return a * (y - b)


def dy_dt(x, y, z):
    return b * x - y - x * z


def dz_dt(x, y, z):
    return x * y - c * z


# initializing x, y, z coordinates
x = np.zeros_like(t)
y = np.zeros_like(t)
z = np.zeros_like(t)

# assigning initial position
x[0] = x0
y[0] = y0
z[0] = z0

# time delta
dt = t[1] - t[0]

# 3D-path (x, y, z coordinates) calculation
for i in range(1, len(t)):
    x[i] = x[i - 1] + dx_dt(x[i - 1], y[i - 1], z[i - 1]) * dt
    y[i] = y[i - 1] + dy_dt(x[i - 1], y[i - 1], z[i - 1]) * dt
    z[i] = z[i - 1] + dz_dt(x[i - 1], y[i - 1], z[i - 1]) * dt

# # Plotting the path
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title('Path of the Bee')
# plt.show()

# 3D - interactive plot
fig = px.line_3d(x=x, y=y, z=z)
fig.show()
