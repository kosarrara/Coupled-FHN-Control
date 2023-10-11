using DifferentialEquations
using Peaks

function kuramoto_order_parameter(x_values, y_values)
    # theta_values = np.arctan2(x_values, y_values)
    # z = np.mean(np.exp(1j*theta_values))
    # return np.abs(z)
    z = mean(exp.(im*atan.(y_values, x_values)))
    return abs(z)
end

function observables(sol)
    t_values = sol.t
    x1_values, y1_values, x2_values, y2_values = [[],[], [], []]
    for coords in sol.u
        push!(x1_values, coords[1])
        push!(y1_values, coords[2])
        push!(x2_values, coords[3])
        push!(y2_values, coords[4])
    end

    x_difference = x1_values - x2_values
    y_difference = y1_values - y2_values
    norm_difference = sqrt.(x_difference.^2 + y_difference.^2)
    peak_indices, peak_values = findmaxima(norm_difference)
    peak_times = t_values[peak_indices]

    return t_values, x1_values, y1_values, x2_values, y2_values, norm_difference, peak_times, peak_values
end

function bc_model(du, u, h, p, t)
    a1, a2, c, eps = p
    past_x1 = h(p, t - tau)[1]
    past_x2 = h(p, t - tau)[3]
    x1, y1, x2, y2 = u
    du[1] = eps * (x1 - (x1^3)/3 - y1 + c*atan(past_x2))
    du[2] = x1 + a1
    du[3] = eps * (x2 - (x2^3)/3 - y2 + c*atan(past_x1))
    du[4] = x2 + a2
end

h(p, t) = [1.0, 0.0, -1.0, 0.0] # History function. For now, it's just the initial condition.
tau = 10.0
lags = [tau]

a1 = 0.7
a2 = 0.7
c = 1.0
eps = 0.01

p = (a1, a2, c, eps, tau)
tspan = (0.0, 5000.0)
u0 = [-a1*3, 0.0, a1*1.0, 0.0]

prob = DDEProblem(bc_model, u0, h, tspan, p; constant_lags = lags, dtmax = 0.1)
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg)
t_values, x1_values, y1_values, x2_values, y2_values, norm_difference, peak_times, peak_values = observables(sol)

print("System solved. Now plotting...")

using Plots
l = @layout [a ; b ; c]
p1 = plot(sol)
p2 = plot(peak_times, peak_values, xlabel="Time", ylabel="Amplitude of difference")
p3 = plot(t_values, norm_difference, xlabel="Time", ylabel="Norm of difference")
plot(p1, p2, p3, layout=l, size=(800, 600))
