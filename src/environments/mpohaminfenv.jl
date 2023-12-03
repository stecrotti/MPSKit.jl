"
    This object manages the hamiltonian environments for an InfiniteMPS
"
mutable struct MPOHamInfEnv{H<:MPOHamiltonian,V,S<:InfiniteMPS,A} <: AbstractInfEnv
    opp::H

    dependency::S
    solver::A

    lw::PeriodicArray{V,1}
    rw::PeriodicArray{V,1}

    lock::ReentrantLock
end

function Base.copy(p::MPOHamInfEnv)
    return MPOHamInfEnv(p.opp, p.dependency, p.solver, copy(p.lw), copy(p.rw))
end

function gen_lw_rw(st::InfiniteMPS, ham::Union{SparseMPO,MPOHamiltonian})
    lw = PeriodicArray(map(st.AL, ham) do al, h
                           V_mps = SumSpace(_firstspace(al))
                           V_mpo = space(h, 1)
                           return BlockTensorMap(undef, scalartype(st),
                                                 V_mps ⊗ V_mpo' ← V_mps)
                       end)
    rw = PeriodicArray(map(st.AR, ham) do ar, h
                           V_mps = SumSpace(_lastspace(ar)')
                           V_mpo = space(h, 4)
                           return BlockTensorMap(undef, scalartype(st),
                                                 V_mps ⊗ V_mpo' ← V_mps)
                       end)

    # lw = PeriodicArray{A,2}(undef, ham.odim, length(st))
    # rw = PeriodicArray{A,2}(undef, ham.odim, length(st))

    # for i in 1:length(st), j in 1:(ham.odim)
    #     lw[j, i] = similar(
    #         st.AL[1], _firstspace(st.AL[i]) * ham[i].domspaces[j]' ← _firstspace(st.AL[i])
    #     )
    #     rw[j, i] = similar(
    #         st.AL[1], _lastspace(st.AR[i])' * ham[i].imspaces[j]' ← _lastspace(st.AR[i])'
    #     )
    # end

    randomize!.(lw)
    randomize!.(rw)

    return lw, rw
end

#randomly initialize envs
function environments(st::InfiniteMPS, ham::MPOHamiltonian, above=nothing;
                      solver=Defaults.linearsolver)
    (isnothing(above) || above === st) ||
        throw(ArgumentError("MPOHamiltonian requires top and bottom states to be equal."))
    lw, rw = gen_lw_rw(st, ham)
    envs = MPOHamInfEnv(ham, similar(st), solver, lw, rw, ReentrantLock())
    return recalculate!(envs, st)
end

function leftenv(envs::MPOHamInfEnv, pos::Int, state)
    check_recalculate!(envs, state)
    return envs.lw[pos]
end

function rightenv(envs::MPOHamInfEnv, pos::Int, state)
    check_recalculate!(envs, state)
    return envs.rw[pos]
end

function recalculate!(envs::MPOHamInfEnv, nstate; tol=envs.solver.tol)
    sameDspace = reduce(&, _lastspace.(envs.lw) .== _firstspace.(nstate.CR))

    if !sameDspace
        envs.lw, envs.rw = gen_lw_rw(nstate, envs.opp)
    end

    solver = envs.solver
    solver = solver.tol == tol ? solver : @set solver.tol = tol
    if Threads.nthreads() > 1
        @sync begin
            Threads.@spawn calclw!(envs.lw, nstate, envs.opp; solver)
            Threads.@spawn calcrw!(envs.rw, nstate, envs.opp; solver)
        end
    else
        calclw!(envs.lw, nstate, envs.opp; solver)
        calcrw!(envs.rw, nstate, envs.opp; solver)
    end

    envs.dependency = nstate
    envs.solver = solver

    return envs
end

function calclw!(fixpoints, st::InfiniteMPS, ham::MPOHamiltonian;
                 solver=Defaults.linearsolver)
    len = length(st)
    @assert len == length(ham)

    # the start element
    leftutil = similar(st.AL[1], left_virtualspace(ham, 1)[1])
    fill_data!(leftutil, one)
    @plansor fixpoints[1][1, 1, 1][-1 -2; -3] = l_LL(st)[-1; -3] * conj(leftutil[-2])
    (len > 1) && left_cyclethrough!(1, fixpoints, ham, st)

    for i in 2:length(left_virtualspace(ham, 1))
        prev = copy(fixpoints[1][1, i, 1]) # use as initial guess in linsolve
        zerovector!(fixpoints[1][1, i, 1])

        left_cyclethrough!(i, fixpoints, ham, st)

        if isone(ham, i) # identity matrices; do the hacky renormalization
            tm = regularize(TransferMatrix(st.AL, st.AL), l_LL(st), r_LL(st))
            fixpoints[1][1, i, 1], convhist = linsolve(flip(tm), fixpoints[1][1, i, 1],
                                                       prev, solver, 1, -1)
            convhist.converged == 0 && @info "calclw failed to converge $(convhist.normres)"

            (len > 1) && left_cyclethrough!(i, fixpoints, ham, st)

            # go through the unitcell, again subtracting fixpoints
            for potato in 1:len
                @plansor fixpoints[potato][i][-1 -2; -3] -= fixpoints[potato][i][1 -2; 2] *
                                                            r_LL(st, potato - 1)[2; 1] *
                                                            l_LL(st, potato)[-1; -3]
            end

        else
            if iszero(ham, i)
                diag = map(b -> b[1, i, i, 1], ham[:])
                tm = TransferMatrix(st.AL, diag, st.AL)
                fixpoints[1][1, i, 1], convhist = linsolve(flip(tm), fixpoints[1][1, i, 1],
                                                           prev, solver, 1, -1)
                convhist.converged == 0 &&
                    @info "calclw failed to converge $(convhist.normres)"
            end
            (len > 1) && left_cyclethrough!(i, fixpoints, ham, st)
        end
    end

    return fixpoints
end

function calcrw!(fixpoints, st::InfiniteMPS, ham::MPOHamiltonian;
                 solver=Defaults.linearsolver)
    len = length(st)
    @assert len == length(ham)
    odim = length(right_virtualspace(ham, len))

    # the start element
    rightutil = similar(st.AL[1], right_virtualspace(ham, len)[end])
    fill_data!(rightutil, one)
    @plansor fixpoints[end][1, end, 1][-1 -2; -3] = r_RR(st)[-1; -3] * conj(rightutil[-2])

    (len > 1) && right_cyclethrough!(odim, fixpoints, ham, st) #populate other sites

    for i in (odim - 1):-1:1
        prev = copy(fixpoints[end][1, i, 1]) # use as initial guess in linsolve
        zerovector!(fixpoints[end][1, i, 1])

        right_cyclethrough!(i, fixpoints, ham, st)

        if isone(ham, i) # identity matrices; do the hacky renormalization
            tm = regularize(TransferMatrix(st.AR, st.AR), l_RR(st), r_RR(st))
            fixpoints[end][1, i, 1], convhist = linsolve(tm, fixpoints[end][1, i, 1], prev,
                                                         solver, 1, -1)
            convhist.converged == 0 && @info "calcrw failed to converge $(convhist.normres)"

            len > 1 && right_cyclethrough!(i, fixpoints, ham, st)

            #go through the unitcell, again subtracting fixpoints
            for potatoe in 1:len
                @plansor fixpoints[potatoe][i][-1 -2; -3] -= fixpoints[potatoe][i][1 -2;
                                                                                   2] *
                                                             l_RR(st, potatoe + 1)[2; 1] *
                                                             r_RR(st, potatoe)[-1; -3]
            end
        else
            if iszero(ham, i)
                diag = map(b -> b[1, i, i, 1], ham[:])
                tm = TransferMatrix(st.AR, diag, st.AR)
                fixpoints[end][1, i, 1], convhist = linsolve(tm, fixpoints[end][1, i, 1],
                                                             prev, solver, 1, -1)
                convhist.converged == 0 &&
                    @info "calcrw failed to converge $(convhist.normres)"
            end

            (len > 1) && right_cyclethrough!(i, fixpoints, ham, st)
        end
    end

    return fixpoints
end
"""
    left_cyclethrough!(index::Int, fp, ham, st)

This function computes all fixpoints at layer index, using the fixpoints at previous layers.
"""
function left_cyclethrough!(index::Int, fp::PeriodicArray{T,1}, ham, st) where {T}
    for i in 1:length(fp)
        zerovector!(fp[i + 1][index])

        transfer = TransferMatrix(st.AL[i], ham[i][1:index, 1, 1, index], st.AL[i])
        fp[i + 1][1, index, 1] = fp[i][1, 1:index, 1] * transfer
        # mul!(fp[i + 1][index], fp[i][1, 1:index, 1], transfer)

        # rmul!(fp[index, i + 1], 0)

        # for j in index:-1:1
        #     contains(ham[i], j, index) || continue

        #     if isscal(ham[i], j, index)
        #         axpy!(
        #             ham.Os[i, j, index],
        #             fp[j, i] * TransferMatrix(st.AL[i], st.AL[i]),
        #             fp[index, i + 1],
        #         )
        #     else
        #         axpy!(
        #             true,
        #             fp[j, i] * TransferMatrix(st.AL[i], ham[i][j, index], st.AL[i]),
        #             fp[index, i + 1],
        #         )
        #     end
        # end
    end
    return nothing
end

function right_cyclethrough!(index::Int, fp::PeriodicArray{T,1}, ham, st) where {T}
    for i in reverse(1:length(fp))
        zerovector!(fp[i - 1][index])

        transfer = TransferMatrix(st.AR[i], ham[i][index, 1, 1, index:end], st.AR[i])
        fp[i - 1][index] = transfer * fp[i][1, index:end, 1]

        # rmul!(fp[index, i - 1], 0)

        # for j in index:size(fp, 1)
        #     contains(ham[i], index, j) || continue

        #     if isscal(ham[i], index, j)
        #         axpy!(
        #             ham.Os[i, index, j],
        #             TransferMatrix(st.AR[i], st.AR[i]) * fp[j, i],
        #             fp[index, i - 1],
        #         )
        #     else
        #         axpy!(
        #             true,
        #             TransferMatrix(st.AR[i], ham[i][index, j], st.AR[i]) * fp[j, i],
        #             fp[index, i - 1],
        #         )
        #     end
        # end
    end
    return nothing
end
