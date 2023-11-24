import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks, hilbert
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x, eps, a):
    return np.array([(x[0] - (x[0]**3)/3 - x[1])/eps, x[0] + a])

def Bmatrix(phi, eps, sigma):
    return sigma*np.array([[np.cos(phi)/eps, np.sin(phi)/eps], [-np.sin(phi), np.cos(phi)]])

def system(t, state, a1, a2, eps, c, sigma):
    x1 = state[0:2]
    x2 = state[2:4]
    phi = np.pi/2 - 0.1
    B = Bmatrix(phi, eps, sigma)
    dx1dt = f(x1, eps, a1) + c * B @ (x2 - x1)
    dx2dt = f(x2, eps, a2) + c * B @ (x1 - x2)
    return np.concatenate((dx1dt, dx2dt))

def kuramoto_order_parameter(x1_values, y1_values, x2_values, y2_values):
    theta1_values = np.angle(x1_values + 1j*hilbert(x1_values)) #np.arctan2(x1_values, y1_values)
    theta2_values = np.angle(x2_values + 1j*hilbert(x2_values)) #np.arctan2(x2_values, y2_values) 
    z = (np.exp(1j*theta1_values) + np.exp(1j*theta2_values))/2
    return np.abs(z)

def synchronization_error(x1_values, y1_values, x2_values, y2_values): # Could be made far more elegant and ready for more neurons if I just input sol.y
    x_mean = (x1_values + x2_values)/2
    y_mean = (y1_values + y2_values)/2
    return np.sqrt((x1_values - x_mean)**2 + (y1_values - y_mean)**2 + (x2_values - x_mean)**2 + (y2_values - y_mean)**2)

def system_observables(a1, a2, eps, c, sigma, initial_state, t_span, max_step=0.01):
    sol = solve_ivp(system, t_span, initial_state, max_step=max_step, args=(a1, a2, eps, c, sigma))

    x1_eq = np.array([-a1, a1**3/3 - a1]) # + correcciones si a1 \neq a2
    x2_eq = np.array([-a2, a2**3/3 - a2])

    t_values = sol.t
    x1_values, y1_values, x2_values, y2_values = sol.y
    x1_from_eq = x1_values - x1_eq[0]
    y1_from_eq = y1_values - x1_eq[1]
    x2_from_eq = x2_values - x2_eq[0]
    y2_from_eq = y2_values - x2_eq[1]
    kuramoto_values = kuramoto_order_parameter(x1_from_eq, y1_from_eq, x2_from_eq, y2_from_eq)
    # x_difference = x1_from_eq - x2_from_eq
    # y_difference = y1_from_eq - y2_from_eq
    # norm_difference = np.sqrt(x_difference**2 + y_difference**2)
    synch_error = synchronization_error(x1_from_eq, y1_from_eq, x2_from_eq, y2_from_eq)
    peak_locations = find_peaks(synch_error)[0]
    peak_times, peak_values = (t_values[peak_locations], synch_error[peak_locations])
    return t_values, x1_values, y1_values, x2_values, y2_values, synch_error, peak_times, peak_values, kuramoto_values

if __name__ == '__main__':

    # Define the parameters:
    a1 = 0.5
    a2 = 0.5
    eps = 0.05
    c = 1
    sigma = 0.05
    animate = False
    # Define the initial condition
    t_transient = 10
    pre_initial_state = [0.0, 0.0, 0.0, 0.0]
    synch_state = solve_ivp(system, (0, t_transient), pre_initial_state, args=(a1, a2, eps, c, sigma)).y[:, -1]
    initial_state = synch_state + 1e-5*np.random.randn(4)
    # Define the time span:
    t_span = (0, 50)

    # Solve the system
    t_values, x1_values, y1_values, x2_values, y2_values, norm_difference, peak_times, peak_values, kuramoto = system_observables(a1, a2, eps, c, sigma, initial_state, t_span)

    fig, (ax1, ax2, ax4) = plt.subplots(nrows=3, ncols=1, figsize=(8, 7))

    ax1.plot(t_values, x1_values, label=r'$u_1(t)$')
    ax1.plot(t_values, y1_values, label=r'$v_1(t)$')
    ax1.plot(t_values, x2_values, label=r'$u_2(t)$')
    ax1.plot(t_values, y2_values, label=r'$v_2(t)$')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Values of state variables')
    ax1.set_title('Solution of the 4D System')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(t_values, norm_difference, label='x difference')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Values')
    ax2.set_title('Synchronization error')
    ax2.grid(True)

    # ax3.plot(peak_times, peak_values, "o-", label='Amplitude of the difference')
    # ax3.set_xlabel('Time (t)')
    # ax3.set_ylabel('Amplitude')
    # ax3.set_title('Amplitude of the difference between the two systems')
    # ax3.grid(True)

    ax4.plot(t_values, kuramoto, label='Kuramoto order parameter')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Kuramoto order parameter')
    ax4.set_title('Kuramoto order parameter')
    # ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True)
    # x_values = np.array([x1_values, x2_values])
    # y_values = np.array([y1_values, y2_values])
    # ax4.plot(t_values, kuramoto_order_parameter(,))

    fig.tight_layout()

    if animate:
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

        skips = 15
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
