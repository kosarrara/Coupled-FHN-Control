using Pkg

dependencies = ["DifferentialEquations",
                "OrdinaryDiffEq",
                "StaticArrays",
                "DynamicalSystems",
                "Plots",
                "LaTeXStrings",
                "ProgressMeter",
                "Peaks"]

using StaticArrays, LinearAlgebra
using DynamicalSystems
using OrdinaryDiffEq
using Plots, LaTeXStrings
using ProgressMeter
using Base.Threads


for dep in dependencies
    Pkg.add(dep)
end
