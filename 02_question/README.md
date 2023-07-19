# Question 02 - plotting path in 3D space

## Problem statement
> Write a program to plot the following equation over time. The equation defines a dynamical system in which the position of the system changes with time. You have to plot the positions over time. Think of this as a Bee moving in 3D space. You need to plot the path taken by the Bee.
> Assume x, y, and z are points in 3D space in which x0 = 0, y0 = 1, z0 = 1.05 and a = 10, b = 28, c = 2.667 are the parameters, and ẋ = (dx/dt), ẏ = (dy/dt), ż = (dz/dt).
> The equations are:

> `ẋ = a * (y-b)`

> `ẏ = b * x - y- x * z`

> `ż = x * y - c * z`

## Solution
* The [Q2-path_in_3D_space.py](https://github.com/kdineshchitra/submission_dhvani_hackathon_computer_vision/blob/master/02_question/Q2-path_in_3D_space.py) holds the plotting script for the given equations.
* The resulting 3D plot will be displayed (in a pop-up window) when executing the [Q2-path_in_3D_space.py](https://github.com/kdineshchitra/submission_dhvani_hackathon_computer_vision/blob/master/02_question/Q2-path_in_3D_space.py) script.

### Explanation
* The script follows the instructions given in question, such as assigning the initial position and parameters, defining the equations.
* The time interval in assumed and the x, y, z - coordinates are updated with dx, dy, dz for each timestep.

### Environment setup
* Requires `Python 3.9+`
* To install the required python libraries run:
* `python -m pip install -r requirements.txt`

### Run script
* To plot the 3D graph:
* `python Q2-path_in_3D_space.py`
