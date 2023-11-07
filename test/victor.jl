using MPSKit
using Test

using TestEnv; TestEnv.activate("MPSKit")

include("setup.jl")

psi = FiniteMPS((ℂ^2)^10, ℂ^4)
H = transverse_field_ising()

envs = environments(psi, H)

ψ, envs, δ = find_groundstate(psi, H)

ψ2, envs = timestep!(ψ, H, 0.1, TDVP(), envs)

H_mpo = make_time_mpo(H, 0.1, TaylorCluster{1}())


excitations(H, QuasiparticleAnsatz(), 0.1, ψ, envs)