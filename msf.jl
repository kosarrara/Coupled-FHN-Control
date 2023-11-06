using StaticArrays
using DynamicalSystems
using OrdinaryDiffEq
using Plots

function fhn_eom(x, params, t)
    a = params[1]
    eps = params[2]
    dx = (x[1] - (x[1]^3)/3 - x[2])/eps
    dy = x[1] + a
    return SVector(dx, dy)
end

function fhn_jac(x, params, t)
    eps = params[2]
    dx_dx = (1 - x[1]^2)/eps
    dx_dy = -1/eps
    dy_dx = 1
    dy_dy = 0
    return SMatrix{2,2}(dx_dx, dx_dy, dy_dx, dy_dy)
end

function msf(xchi, params, t)
    aeps = params[1:2]
    alpha = params[3]
    beta = params[4]
    c = params[5] # Coupling strength
    B = Bmatrix(pi/2, aeps[2])
    x = xchi[1:2]
    chireal = xchi[3:4]
    chiimag = xchi[5:6]
    dxdy = fhn_eom(x, aeps, t)
    dchireal = fhn_jac(x, aeps, t) * chireal + c*(alpha*B*chireal - beta*B*chiimag)
    dchiimag = fhn_jac(x, aeps, t) * chiimag + c*(alpha*B*chiimag + beta*B*chireal)
    return [dxdy; dchireal; dchiimag]
end

function Bmatrix(phi, eps)
    return SA[cos(phi)/eps -sin(phi)/eps; sin(phi) cos(phi)] # Revisar si esta es efectivamente B
end

diffeq = (alg=Tsit5(), abstol = 1e-9, reltol = 1e-9)
# fhn_system = ContinuousDynamicalSystem(fhn_eom, SA[1.3, 0], SA[0.5, 0.1])
msf_system(alpha, beta) = ContinuousDynamicalSystem(msf, SA[1, 1, 0.0, 0.0, 0.0, 0.0], SA[0.5, 0.01, alpha, beta, -1.0], diffeq=diffeq)

test = msf_system(0.3, 0.3)
traj, _ = trajectory(test, 100.0; Δt = 0.001)
λ = lyapunov(test, 200.0; Δt = 0.001, Ttr = 100.0)
println(λ)
plot(traj[:, 1], traj[:, 2])
