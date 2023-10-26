using DifferentialEquations
using Peaks
using LaTeXStrings

function kuramoto_order_parameter(x_values, y_values)
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

function fn_model(du, u, h, p, t)
    a1, a2, c, eps, crit_x = p
    past_x1 = h(p, t - tau)[1]
    past_x2 = h(p, t - tau)[3]
    x1, y1, x2, y2 = u
    du[1] = (x1 - (x1^3)/3 - y1 + c*(past_x2 - x1)*sign(x1 - crit_x))/eps#atan(past_x2 - x1))/eps
    du[2] = x1 + a1
    du[3] = (x2 - (x2^3)/3 - y2 + c*(past_x1 - x2)*sign(x2 - crit_x))/eps#atan(past_x1 - x2))/eps
    du[4] = x2 + a2
end

tau = 2.5
lags = [tau]
a1 = 1.3
a2 = 1.3
c = 1.0
eps = 0.01
crit_x = 0.5

p = (a1, a2, c, eps, tau, crit_x)
tspan = (0.0, 1000.0)
u0 = [2.1, 0, 2.2, 0]#[-a1*2, 0.0, a1*0.0, 0.0]

function h(p, t)
    if t > -0.5
        return [2, 0, 2, 0] # History function. For now, it's just the initial condition.
    else
        return [-a1, 0, -a1, 0]
    end
end

prob = DDEProblem(fn_model, u0, h, tspan, p; constant_lags = lags, dtmax = 0.01)
alg = MethodOfSteps(Tsit5())
sol = solve(prob, alg, maxiters=1e8)
t_values, x1_values, y1_values, x2_values, y2_values, norm_difference, peak_times, peak_values = observables(sol)

print("System solved. Now plotting...")

using Plots
l = @layout [a ; b ; c]
p1 = plot(sol, xlabel="Time", ylabel="System Variables", labels=[L"$u_1$" L"$v_1$" L"$u_2$" L"$v_2$"], dpi=600)
p2 = plot(t_values, norm_difference, xlabel="Time", ylabel="Norm of difference")
p3 = plot(peak_times, peak_values, xlabel="Time", ylabel="Amplitude of difference")
p = plot(p1, p2, p3, layout=l, size=(800, 600))
gui(p)