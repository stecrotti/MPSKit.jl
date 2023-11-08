#https://arxiv.org/pdf/1901.05824.pdf

@kwdef struct WII <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
end

@kwdef struct TaylorCluster <: Algorithm
    N::Int = 2
    extension::Bool = true
    compression::Bool = true
end

const WI = TaylorCluster(; N=1, extension=false, compression=false)

function make_time_mpo(H::MPOHamiltonian, dt::Number, alg::TaylorCluster)
    N = alg.N
    τ = -1im * dt
    # start with H^N
    H_n = H^N
    linds = LinearIndices(ntuple(i -> virtualdim(H), N))
    cinds = CartesianIndices(linds)
    
    # extension step: Algorithm 3
    # incorporate higher order terms
    # TODO: don't need to fully construct H_next...
    if alg.extension
        H_next = H_n * H
        linds_next = LinearIndices(ntuple(i -> virtualdim(H), N + 1))
        for (i, slice) in enumerate(H_n.data)
            for a in cinds, b in cinds
                all(>(1), b.I) || continue
                all(in((1, virtualdim(H))), a.I) && any(==(virtualdim(H)), a.I) && continue
                
                n1 = count(==(1), a.I) + 1
                n3 = count(==(virtualdim(H)), b.I) + 1
                factor = τ * factorial(N) / (factorial(N + 1) * n1 * n3)
                
                for c in 1:N+1, d in 1:N+1
                    aₑ = insert!([a.I...], c, 1)
                    bₑ = insert!([b.I...], d, virtualdim(H))
                    
                    # TODO: use VectorInterface for memory efficiency
                    slice[linds[a], 1, 1, linds[b]] += factor * H_next[i][linds_next[aₑ...], 1, 1, linds_next[bₑ...]]
                end
            end
        end
    end
    
    # loopback step: Algorithm 1
    # constructing the Nth order time evolution MPO
    mpo = convert(SparseMPO, H_n)
    for slice in mpo.data
        for b in cinds[2:end]
            all(in((1, virtualdim(H))), b.I) || continue
            
            b_lin = linds[b]
            a = count(==(virtualdim(H)), b.I)
            factor = τ^a * factorial(N - a) / factorial(N)
            slice[:, 1, 1, 1] = slice[:, 1, 1, 1] + factor * slice[:, 1, 1, b_lin]
            for I in BlockTensorKit.nonzero_keys(slice)
                (I[1] == b_lin || I[4] == b_lin) && delete!(slice, I)
            end
        end
    end
    
    # Remove equivalent rows and columns: Algorithm 2
    for slice in mpo.data
        for c in cinds
            c_lin = linds[c]
            s_c = CartesianIndex(sort(c.I, by=(!=(1))))
            s_r = CartesianIndex(sort(c.I, by=(!=(virtualdim(H)))))
            
            n1 = count(==(1), c.I)
            n3 = count(==(virtualdim(H)), c.I)
            
            if n3 <= n1 && s_c != c
                slice[linds[s_c], 1, 1, :] += slice[c_lin, 1, 1, :]
                for I in BlockTensorKit.nonzero_keys(slice)
                    (I[1] == c_lin || I[4] == c_lin) && delete!(slice, I)
                end
            elseif n3 > n1 && s_r != c
                slice[:, 1, 1, linds[s_r]] += slice[:, 1, 1, c_lin]
                for I in BlockTensorKit.nonzero_keys(slice)
                    (I[1] == c_lin || I[4] == c_lin) && delete!(slice, I)
                end
            end
        end
    end
    
    # Approximate compression step: Algorithm 4
    if alg.compression
        for slice in mpo.data
            for a in cinds
                all(>(1), a.I) || continue
                a_lin = linds[a]
                n1 = count(==(virtualdim(H)), a.I)
                b = CartesianIndex(replace(a.I, virtualdim(H) => 1))
                b_lin = linds[b]
                factor = τ^n1 * factorial(N - n1) / factorial(N)
                slice[:, 1, 1, b_lin] += factor * slice[:, 1, 1, a_lin]
                
                for I in BlockTensorKit.nonzero_keys(slice)
                    (I[1] == a_lin || I[4] == a_lin) && delete!(slice, I)
                end
            end
        end
    end
    
    return remove_orphans!(mpo)
end

function make_time_mpo(ham::MPOHamiltonian{T}, dt, alg::WII) where {T}
    WA = PeriodicArray{T,3}(undef, ham.period, ham.odim - 2, ham.odim - 2)
    WB = PeriodicArray{T,2}(undef, ham.period, ham.odim - 2)
    WC = PeriodicArray{T,2}(undef, ham.period, ham.odim - 2)
    WD = PeriodicArray{T,1}(undef, ham.period)

    δ = dt * (-1im)

    for i in 1:(ham.period), j in 2:(ham.odim - 1), k in 2:(ham.odim - 1)
        init_1 = isometry(
            storagetype(ham[i][1, ham.odim]),
            codomain(ham[i][1, ham.odim]),
            domain(ham[i][1, ham.odim]),
        )
        init = [init_1, zero(ham[i][1, k]), zero(ham[i][j, ham.odim]), zero(ham[i][j, k])]

        (y, convhist) = exponentiate(
            1.0, RecursiveVec(init), Arnoldi(; tol=alg.tol, maxiter=alg.maxiter)
        ) do x
            out = similar(x.vecs)

            @plansor out[1][-1 -2; -3 -4] :=
                δ * x[1][-1 1; -3 -4] * ham[i][1, ham.odim][2 3; 1 4] * τ[-2 4; 2 3]

            @plansor out[2][-1 -2; -3 -4] :=
                δ * x[2][-1 1; -3 -4] * ham[i][1, ham.odim][2 3; 1 4] * τ[-2 4; 2 3]
            @plansor out[2][-1 -2; -3 -4] +=
                sqrt(δ) * x[1][1 2; -3 4] * ham[i][1, k][-1 -2; 3 -4] * τ[3 4; 1 2]

            @plansor out[3][-1 -2; -3 -4] :=
                δ * x[3][-1 1; -3 -4] * ham[i][1, ham.odim][2 3; 1 4] * τ[-2 4; 2 3]
            @plansor out[3][-1 -2; -3 -4] +=
                sqrt(δ) * x[1][1 2; -3 4] * ham[i][j, ham.odim][-1 -2; 3 -4] * τ[3 4; 1 2]

            @plansor out[4][-1 -2; -3 -4] :=
                δ * x[4][-1 1; -3 -4] * ham[i][1, ham.odim][2 3; 1 4] * τ[-2 4; 2 3]
            @plansor out[4][-1 -2; -3 -4] +=
                x[1][1 2; -3 4] * ham[i][j, k][-1 -2; 3 -4] * τ[3 4; 1 2]
            @plansor out[4][-1 -2; -3 -4] +=
                sqrt(δ) * x[2][1 2; -3 -4] * ham[i][j, ham.odim][-1 -2; 3 4] * τ[3 4; 1 2]
            @plansor out[4][-1 -2; -3 -4] +=
                sqrt(δ) * x[3][-1 4; -3 3] * ham[i][1, k][2 -2; 1 -4] * τ[3 4; 1 2]

            RecursiveVec(out)
        end
        convhist.converged == 0 && @warn "failed to exponentiate $(convhist.normres)"

        WA[i, j - 1, k - 1] = y[4]
        WB[i, j - 1] = y[3]
        WC[i, k - 1] = y[2]
        WD[i] = y[1]
    end

    W2 = PeriodicArray{Union{T,Missing},3}(missing, ham.period, ham.odim - 1, ham.odim - 1)
    W2[:, 2:end, 2:end] = WA
    W2[:, 2:end, 1] = WB
    W2[:, 1, 2:end] = WC
    W2[:, 1, 1] = WD

    return SparseMPO(W2)
end
