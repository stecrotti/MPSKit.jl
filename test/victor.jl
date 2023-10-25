using MPSKit
using Test

include("setup.jl")

psi = InfiniteMPS(2, 12)
H = transverse_field_ising()

ψ, envs, δ = find_groundstate(psi, H)

excitations(H, QuasiparticleAnsatz(), 0.1, ψ, envs)