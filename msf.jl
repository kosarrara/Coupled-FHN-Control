using StaticArrays, LinearAlgebra
using DynamicalSystems
using OrdinaryDiffEq
using Plots, LaTeXStrings
using ProgressMeter
using Base.Threads

gr()

# Problemas pendientes:
# 1. El lyapunov parece ser más grande de lo que aparece en el paper. Las
#    regiones de estabilidad están piola eso si.
# 2. Todavía no puedo hacer esos gráficos bonitos del paper.
# 3. Hay que saber cómo cambiaría de acuerdo a los parámetros la región de estabilidad.
#    Preguntarle a Javier.

# Parametros que me falta fijar:
# 1. El c de acople no sé si está igual que el del paper.
# 2. El tiempo de cálculo del Lyapunov.
# 3. El tiempo transitorio de Lyapunov.
# 4. La perturbación para calcular el Lyapunov.

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
    dchireal = fhn_jac(x, aeps, t) * chireal + c*(alpha*B*chireal - beta*B*chiimag)
    dchiimag = fhn_jac(x, aeps, t) * chiimag + c*(alpha*B*chiimag + beta*B*chireal)
    return [dxdy; dchireal; dchiimag]
end

function couplingJacobian(phi, eps)
    return SA[cos(phi)/eps sin(phi)/eps; -sin(phi) cos(phi)]
end

function msf_system(alpha, beta; a=0.5, eps=0.05, coupling=1.0, phi=(pi/2)-0.1, diffeq=(alg=Tsit5(), abstol = 1e-9, reltol = 1e-9))
    ds = ContinuousDynamicalSystem(msf_eom, SA[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], SA[a, eps, alpha, beta, coupling, phi], diffeq=diffeq)
    return ds
end

function master_stability_function(alpha, beta; testfunc=(state1, d0) -> [state1[1:2] ; state1[3:end] .+ d0/sqrt(4)], kwargs...)
    system = msf_system(alpha, beta; kwargs...)
    return lyapunov(system, 100.0; Δt = 0.1, Ttr = 10.0, inittest=testfunc, d0=1e-9)
end

function plot_msf_regions(n_rows; kwargs...)
    alpha_sweep = range(-0.5, 1.5, length=n_rows)
    beta_sweep = range(-0.5, 0.5, length=n_rows)
    msf = zeros(length(alpha_sweep), length(beta_sweep))

    @showprogress for j in 1:length(alpha_sweep)
        alpha = alpha_sweep[j]
        Threads.@threads for i in 1:length(beta_sweep)
            beta = beta_sweep[i]
            msf[i, j] = master_stability_function(alpha, beta; kwargs...)
        end
    end

    levels = [-1e10, 0, 1e10]

    p = contour(alpha_sweep, beta_sweep, msf;
                levels=levels,
                fill=true,
                xlabel=L"α",
                ylabel=L"β",
                zlabel=L"λ",
                # lw=1,
                # line_smoothing=0.85,
                # clabels=true,
                # cbar=false,
                # color=:plasma
                )
    display(p)
end

function plot_msf_vs_eigs(start, stop, n_points, kwargs...)
    eigenvalue_real_sweep = range(start, stop, length=n_points)
    msf_sweep = zeros(length(eigenvalue_real_sweep))

    Threads.@threads for i in 1:length(eigenvalue_real_sweep)
        msf_sweep[i] = master_stability_function(eigenvalue_real_sweep[i], 0.0; kwargs...)
    end

    p = plot(eigenvalue_real_sweep, msf_sweep;
              xlabel="Eigenvalue",
              ylabel="MSF(Eigenvalue)"
              )
    display(p)
end

function synch_is_stable(coupling_matrix)
    eigenvalues = eigvals(coupling_matrix)
    msf_for_eigs = master_stability_function.(real.(eigenvalues), imag.(eigenvalues))
    return all(msf_for_eigs .< 0)
end