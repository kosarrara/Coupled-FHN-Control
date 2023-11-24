using StaticArrays, LinearAlgebra
using DynamicalSystems
using OrdinaryDiffEq
using Plots, LaTeXStrings
using ProgressMeter
using Base.Threads

gr()

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
    returnable = SA_F64[dx_dx dx_dy; dy_dx dy_dy] # Revisar con más detalle

    return returnable
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

function master_stability_function(alpha, beta; testfunc=(state1, d0) -> [state1[1:2] ; state1[3:end] .- d0/sqrt(4)], kwargs...)
    system = msf_system(alpha, beta; kwargs...)
    return lyapunov(system, 100.0; Δt = 0.1, Ttr = 100.0, inittest=testfunc, d0=1e-8)
end

function plot_msf_regions(n_rows; kwargs...)
    alpha_sweep = range(-1.5, 0.5, length=n_rows)
    beta_sweep = range(-1.5, 1.5, length=n_rows)
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
                cbar=false
                )
    display(p)
end

function plot_msf_vs_eigs(start, stop, n_points; kwargs...)
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

function synch_is_stable(coupling_matrix; kwargs...)
    eigenvalues = eigvals(coupling_matrix)
    msf_for_eigs = master_stability_function.(real.(eigenvalues), imag.(eigenvalues); kwargs...)
    return all(msf_for_eigs .≤ 0)
end

function ring_coupling(size; sigma=1, neighbors=1)
    coupling_matrix = zeros(size, size)
    # coupling_matrix[1, end] = 1
    # coupling_matrix[end, 1] = 1    
    if size > 2*neighbors
        correction = -2*neighbors
    else
        correction = -size + 1
    end
    # coupling_matrix[1, 1] = correction
    # coupling_matrix[end, end] = -1
    for i in 1:size
        if i + neighbors ≤ size
            coupling_matrix[i, i .+ (1:neighbors)] .+= 1
            coupling_matrix[i .+ (1:neighbors), i] .+= 1
            coupling_matrix[i, i] = correction
        else
            wrap = i+neighbors - size
            coupling_matrix[i, i+1:end] .+= 1
            coupling_matrix[i, 1:wrap] .+= 1
            coupling_matrix[i+1:end, i] .+= 1
            coupling_matrix[1:wrap, i] .+= 1
            coupling_matrix[i, i] = correction
        end
    end
    return sigma.*coupling_matrix
end

function plot_msf_regions_with_eigs(n_rows, coupling_matrix; savefigure=false, kwargs...)
    alpha_sweep = range(-1.5, 0.5, length=n_rows)
    beta_sweep = range(-0.5, 0.5, length=n_rows)
    msf = zeros(length(alpha_sweep), length(beta_sweep))

    @showprogress for j in 1:length(alpha_sweep)
        alpha = alpha_sweep[j]
        Threads.@threads for i in 1:length(beta_sweep)
            beta = beta_sweep[i]
            msf[i, j] = master_stability_function(alpha, beta; kwargs...)
        end
    end

    eigs = eigvals(coupling_matrix)
    msf_for_eigs = master_stability_function.(real.(eigs), imag.(eigs); kwargs...)
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
                cbar=false,
                xlims=(-1.5, 0.5),
                ylims=(-0.5, 0.5)
                )
    plot!(p, real.(eigs), imag.(eigs), seriestype=:scatter, color=:red, label="Eigenvalues", dpi=400, aspect_ratio=2)

    if savefigure
        savefig(p, "msf_with_eigs.png")
    end
    display(p)
    if all(msf_for_eigs .≤ 0)
        println("Synchronization is stable")
        println("Max Lyapunov: ", maximum(msf_for_eigs))
    else
        println("Synchronization is unstable")
        println("Max Lyapunov: ", maximum(msf_for_eigs))
    end
end

function get_crit_coupling(coupling_matrix)
    coupling = 0.01
    while !synch_is_stable(coupling_matrix; coupling=coupling)
        coupling += 0.01
    end
    return coupling - 0.005
end