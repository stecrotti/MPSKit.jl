"
    FinEnv keeps track of the environments for FiniteMPS / WindowMPS
    It automatically checks if the queried environment is still correctly cached and if not - recalculates

    if above is set to nothing, above === below.

    opp can be a vector of nothing, in which case it'll just be the overlap
"
struct FinEnv{A,B,C,D} <: Cache
    above::A

    opp::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{D}
    rightenvs::Vector{D}
end

function environments(below, t::Tuple, args...; kwargs...)
    return environments(below, t[1], t[2], args...; kwargs...)
end
function environments(below, opp, leftstart, rightstart)
    return environments(below, opp, nothing, leftstart, rightstart)
end
function environments(below, opp, above, leftstart, rightstart)
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(below)
        push!(leftenvs, similar(leftstart))
        push!(rightenvs, similar(rightstart))
    end
    t = similar(below.AL[1])
    return FinEnv(
        above,
        opp,
        fill(t, length(below)),
        fill(t, length(below)),
        leftenvs,
        reverse(rightenvs),
    )
end

#automatically construct the correct leftstart/rightstart for a finitemps
function environments(
    below::FiniteMPS{S}, ham::Union{SparseMPO,MPOHamiltonian}, above=nothing
) where {S}
    GL = map(1:length(below)) do i
        Vmps = SumSpace(left_virtualspace(below, i))
        Vmpo = left_virtualspace(ham, i)
        return TensorMap(undef, scalartype(below), Vmps ⊗ Vmpo', Vmps)
    end
    
    GR = map(1:length(below)) do i
        Vmps = SumSpace(right_virtualspace(below, i))
        Vmpo = right_virtualspace(ham, i)
        return TensorMap(undef, scalartype(below), Vmps ⊗ Vmpo', Vmps)
    end
    
    util_left = Tensor(undef, scalartype(below), space(GL[1][1], 2))
    @tensor GL[1][1][-1 -2; -3] := l_LL(below)[-1; -3] * util_left[-2]
    
    idx = ham isa SparseMPO ? 1 : lastindex(GR[end])
    util_right = Tensor(undef, scalartype(below), space(GR[end][idx], 2))
    @tensor GR[end][idx][-1 -2; -3] := r_RR(below)[-1; -3] * util_right[-2]
    
    
    left_deps = fill(similar(below.AL[1]), length(below))
    right_deps = fill(similar(below.AR[1]), length(below))
    
    return FinEnv(above, ham, left_deps, right_deps, GL, GR)
end

#extract the correct leftstart/rightstart for WindowMPS
function environments(
    state::WindowMPS,
    ham::Union{SparseMPO,MPOHamiltonian,DenseMPO},
    above=nothing;
    lenvs=environments(state.left_gs, ham),
    renvs=environments(state.right_gs, ham),
)
    return environments(
        state,
        ham,
        above,
        copy(leftenv(lenvs, 1, state.left_gs)),
        copy(rightenv(renvs, length(state), state.right_gs)),
    )
end

function environments(below::S, above::S) where {S<:Union{FiniteMPS,WindowMPS}}
    S isa WindowMPS &&
        (above.left_gs == below.left_gs || throw(ArgumentError("left gs differs")))
    S isa WindowMPS &&
        (above.right_gs == below.right_gs || throw(ArgumentError("right gs differs")))

    opp = fill(nothing, length(below))
    return environments(below, opp, above, l_LL(above), r_RR(above))
end

function environments(state::Union{FiniteMPS,WindowMPS}, opp::ProjectionOperator)
    @plansor leftstart[-1; -2 -3 -4] := l_LL(opp.ket)[-3; -4] * l_LL(opp.ket)[-1; -2]
    @plansor rightstart[-1; -2 -3 -4] := r_RR(opp.ket)[-1; -2] * r_RR(opp.ket)[-3; -4]
    return environments(state, fill(nothing, length(state)), state, leftstart, rightstart)
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::FinEnv, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(cache::FinEnv, ind, ψ)
    a = findlast(i -> ψ.AR[i] !== cache.rdependencies[i], (ind + 1):length(ψ))
    
    if !isnothing(a) # we need to recalculate
        for j in a:-1:(ind + 1)
            above = isnothing(cache.above) ? ψ.AR[j] : cache.above.AR[j]
            cache.rightenvs[j] =
                TransferMatrix(above, cache.opp[j], ψ.AR[j]) * cache.rightenvs[j + 1]
            cache.rdependencies[j] = ψ.AR[j]
        end
    end

    return cache.rightenvs[ind + 1]
end

function leftenv(cache::FinEnv, ind, ψ)
    a = findfirst(i -> ψ.AL[i] !== cache.ldependencies[i], 1:(ind - 1))

    if !isnothing(a) # we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(cache.above) ? ψ.AL[j] : cache.above.AL[j]
            cache.leftenvs[j + 1] =
                cache.leftenvs[j] * TransferMatrix(above, cache.opp[j], ψ.AL[j])
            cache.ldependencies[j] = ψ.AL[j]
        end
    end

    return cache.leftenvs[ind]
end
