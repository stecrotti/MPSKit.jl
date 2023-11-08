function approximate(state, toapprox, alg::Union{DMRG,DMRG2}, envs...)
    return approximate!(copy(state), toapprox, alg, envs...)
end

function approximate!(init::AbstractFiniteMPS, sq, alg, envs=environments(init, sq))
    tor = approximate!(init, [sq], alg, [envs])
    return (tor[1], tor[2][1], tor[3])
end

function approximate!(
    init::AbstractFiniteMPS,
    squash::Vector,
    alg::DMRG2,
    envs=[environments(init, sq) for sq in squash],
)
    t₀ = Base.time_ns()
    ϵ::Float64 = 2 * alg.tol
    for iter in 1:(alg.maxiter)
        ϵ = 0.0
        Δt = @elapsed begin
            for pos in [1:(length(init) - 1); (length(init) - 2):-1:1]
                ac2 = init.AC[pos] * _transpose_tail(init.AR[pos + 1])

                nac2 = sum(
                    map(zip(squash, envs)) do (sq, pr)
                        ac2_proj(pos, init, pr)
                    end,
                )

                (al, c, ar) = tsvd!(nac2; trunc=alg.trscheme)

                ϵ = max(ϵ, norm(al * c * ar - ac2) / norm(ac2))

                init.AC[pos] = (al, complex(c))
                init.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end
            
            
            init, envs =
                alg.finalize(iter, init, squash, envs)::Tuple{typeof(init),typeof(envs)}
        end
    
        alg.verbose && @info "DMRG2 iteration:" iter ϵ Δt
        ϵ <= alg.tol && break
        
        iter == alg.maxiter && @warn "DMRG2 maximum iterations" iter ϵ
    end
    
    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbose && @info "DMRG summary:" ϵ Δt
    return init, envs, ϵ
end

function approximate!(
    init::AbstractFiniteMPS,
    squash::Vector,
    alg::DMRG,
    envs=[environments(init, sq) for sq in squash],
)
    t₀ = Base.time_ns()
    ϵ::Float64 = 2 * alg.tol
    for iter in 1:(alg.maxiter)
        ϵ = 0.0
        Δt = @elapsed begin
            for pos in [1:(length(init) - 1); length(init):-1:2]
                newac = sum(
                    map(zip(squash, envs)) do (sq, pr)
                        ac_proj(pos, init, pr)
                    end,
                )

                ϵ = max(ϵ, norm(newac - init.AC[pos]) / norm(newac))
                init.AC[pos] = newac
            end
            init, envs =
                alg.finalize(iter, init, squash, envs)::Tuple{typeof(init),typeof(envs)}
        end
        
        alg.verbose && @info "DMRG iteration:" iter ϵ Δt

        ϵ <= alg.tol && break
        iter == alg.maxiter && @warn "DMRG maximum iterations" iter ϵ
    end
    
    Δt = (Base.time_ns() - t₀) / 1.0e9
    alg.verbose && @info "DMRG summary:" ϵ Δt
    return init, envs, ϵ
end
