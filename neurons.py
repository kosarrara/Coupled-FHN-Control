import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the parameters:
a = 1.145
eps = 1.0
c = 1.0

# Define the initial conditions:
initial_state = [-a*3, 0.0, a*1.0, 0.0]

# Define the time span:
t_span = (0, 500)

def f(xi, xj):
    return c*np.arctan(xi)
    #return c*(xi-xj)

def system(t, state):
    x1, y1, x2, y2 = state
    dx1dt = eps * (x1 - (x1**3)/3 - y1 + f(x2, x1))
    dy1dt = x1 - a
    dx2dt = eps * (x2 - (x2**3)/3 - y2 + f(x1, x2))
    dy2dt = x2 + a
    return [dx1dt, dy1dt, dx2dt, dy2dt]



sol = solve_ivp(system, t_span, initial_state, method='RK45', max_step=0.05)

t_values = sol.t
x1_values, y1_values, x2_values, y2_values = sol.y

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))

ax1.plot(t_values, x1_values, label='x1(t)')
ax1.plot(t_values, y1_values, label='y1(t)')
ax1.plot(t_values, x2_values, label='x2(t)')
ax1.plot(t_values, y2_values, label='y2(t)')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Values')
ax1.set_title('Solution of the 4D System')
ax1.legend()
ax1.grid(True)

ax2.plot(t_values, x1_values - x2_values, label='x difference')
ax2.plot(t_values, y1_values - y2_values, label='y difference')
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Values')
ax2.set_title('Difference between the two systems')
ax2.grid(True)

fig.tight_layout()

# Create the animation figure
fig_anim, ax_anim = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

for i, ax in enumerate(ax_anim):
    ax.set_xlim(min(min(x1_values), min(x2_values)),
                    max(max(x1_values), max(x2_values)))
    ax.set_ylim(min(min(y1_values), min(y2_values)),
                    max( max(y1_values), max(y2_values)))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'System {i+1}')

# Initialize empty lines for animation
line1_anim, = ax_anim[0].plot([], [], label='System 1', color='blue')
line2_anim, = ax_anim[1].plot([], [], label='System 2', color='orange')
dot1_anim, = ax_anim[0].plot([], [], 'o', color='blue')
dot2_anim, = ax_anim[1].plot([], [], 'o', color='orange')

# Function to update the animation frames

skips = 20
def update(frame):
    frame = skips*frame
    line1_anim.set_data(x1_values[:frame], y1_values[:frame])
    line2_anim.set_data(x2_values[:frame], y2_values[:frame])
    dot1_anim.set_data(x1_values[frame], y1_values[frame])
    dot2_anim.set_data(x2_values[frame], y2_values[frame])
    return line1_anim, line2_anim, dot1_anim, dot2_anim

# Create the animation
ani = FuncAnimation(fig_anim, update, frames=int(len(t_values)/skips), blit=True, interval=1)

# Show the animation figure
plt.show()
