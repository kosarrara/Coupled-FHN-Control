using StaticArrays
using DynamicalSystems
using OrdinaryDiffEq
using Plots
using LaTeXStrings
using ProgressMeter
using Base.Threads

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

function msf_eom(xchi, params, t)
    aeps = params[1:2]
    alpha = params[3]
    beta = params[4]
    c = params[5] 
    phi = params[6]
    B = couplingJacobian(phi, aeps[2])
    x = xchi[1:2]
    chireal = xchi[3:4]
    chiimag = xchi[5:6]
    dxdy = fhn_eom(x, aeps, t)
    dchireal = fhn_jac(x, aeps, t) * chireal - c*(alpha*B*chireal - beta*B*chiimag)
    dchiimag = fhn_jac(x, aeps, t) * chiimag - c*(alpha*B*chiimag + beta*B*chireal)
    return [dxdy; dchireal; dchiimag]
end

function couplingJacobian(phi, eps)
    return SA[cos(phi)/eps sin(phi)/eps; -sin(phi) cos(phi)]
#    return SA[1/eps 0; 0 0] # para ver que onda acá (no da negativo cuando debería).
end

# fhn_system = ContinuousDynamicalSystem(fhn_eom, SA[1.3, 0], SA[0.5, 0.1])
function msf_system(alpha, beta; a=0.5, eps=0.05, coupling=1.0, phi=pi/2 - 0.1, diffeq=(alg=Tsit5(), abstol = 1e-9, reltol = 1e-9))
    ds = ContinuousDynamicalSystem(msf_eom, SA[1.0, 1.0, 0.0, 0.0, 0.0, 0.0], SA[a, eps, alpha, beta, coupling, phi], diffeq=diffeq)
    return ds
end

# test = msf_system(0, 0)
# traj, _ = trajectory(test, 10.0; Δt = 0.001)
# inittest_default(D) = (state1, d0) -> [state1[1:2] ; state1[3:end] .- d0/sqrt(D)] # Para que las perturbaciones sean en el espacio de chi y no en el del x sincronizado.
# λ = lyapunov(test, 2000.0; Δt = 0.01, Ttr = 100.0, inittest = inittest_default(dimension(test)-2))

function master_stability_function(alpha, beta, testfunc; kwargs...)
    system = msf_system(alpha, beta; kwargs...)
    return lyapunov(system, 300.0; Δt = 1.0, Ttr = 100.0, inittest=testfunc, show_progress=true)
end

function main(n_rows)
    alpha_sweep = range(-1.5, 1.5, length=n_rows)
    beta_sweep = range(-1.5, 1.5, length=n_rows)
    msf = zeros(length(alpha_sweep), length(beta_sweep))

    @showprogress for i in 1:length(alpha_sweep)
        alpha = alpha_sweep[i]
        Threads.@threads for j in 1:length(beta_sweep)
            beta = beta_sweep[j]
            msf[i, j] = master_stability_function(alpha, beta, (state1, d0) -> [state1[1:2] ; state1[3:end] .- d0/sqrt(4)]; coupling=1/12)
        end
    end
    levels = [-1e10, 0, 1e10]
    p = contour(beta_sweep, alpha_sweep, msf, levels=levels, fill=true, xlabel = L"\beta", ylabel = L"\alpha", zlabel = L"\lambda", lw=0, line_smoothing=0.85)
    display(p)
end

