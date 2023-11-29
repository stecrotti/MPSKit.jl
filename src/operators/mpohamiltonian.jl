"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian

    really just a sparsempo, with some garantuees on its structure
"


struct MPOHamiltonian{T<:SparseMPOTensor} <: AbstractMPO
    data::PeriodicVector{T}
    function MPOHamiltonian{T}(data::PeriodicVector{T}) where {T<:SparseMPOTensor}
        for i in eachindex(data)
            Vₗ = left_virtualspace(data[i])
            Vᵣ = dual(right_virtualspace(data[i - 1]))
            Vₗ == Vᵣ ||
                throw(SpaceMismatch("Incompatible virtual spaces at $i:\n$Vₗ ≠ $Vᵣ"))
            space(data[i], 2) == dual(space(data[i], 3)) ||
                throw(TensorKit.SpaceMismatch("Incompatible physical spaces at $i"))
        end
        return new{T}(data)
    end
end

MPOHamiltonian(data::AbstractVector{T}) where {T} = MPOHamiltonian{T}(PeriodicArray(data))

# BlockTensorKit.blocktype(::MPOHamiltonian{T}) where {T} = blocktype(T)

physicalspace(H::MPOHamiltonian, i::Int) = physicalspace(H[i])
physicalspace(H::MPOHamiltonian) = mapfoldl(only ∘ physicalspace, ⊗, H.data)

function MPOHamiltonian(t::TensorMap{S,N,N}) where {S,N}
    V₀ = oneunit(S)
    P = space(t, 1)
    if N > 1
        @assert all(isequal(P), space.(Ref(t), 2:N)) "all physical spaces should be equal"
    end
    
    τ = TensorKit.BraidingTensor{S,TensorKit.storagetype(t)}(P, V₀)
    localmpo = decompose_localmpo(add_util_leg(t))
    
    ttype = Union{eltype(localmpo),typeof(τ)}
    
    Vₗ = push!(left_virtualspace.(localmpo), dual(right_virtualspace(localmpo[end])))
    cod = SumSpace(Vₗ) ⊗ P
    Vᵣ = pushfirst!(dual.(right_virtualspace.(localmpo)), left_virtualspace(localmpo[1]))
    dom = P ⊗ SumSpace(Vᵣ)
    
    W = BlockTensorMap{S,2,2,ttype}(undef, cod, dom)
    W[1, 1, 1, 1] = τ
    W[end, 1, 1, end] = τ
    
    for (i, O) in enumerate(localmpo)
        W[i, 1, 1, i + 1] = O
    end
    
    return MPOHamiltonian(PeriodicArray([W]))
end

Base.parent(H::MPOHamiltonian) = H.data

function Base.show(io::IO, ::MIME"text/plain", H::Union{MPOHamiltonian,SparseMPO})
    typestr = H isa MPOHamiltonian ? "MPOHamiltonian" : "SparseMPO"
    print(io, "$(length(H))-periodic $(typestr):")
    print(io, " ⋯ ")
    join(io, physicalspace(H), " ⊗ ")
    println(io, " ⋯")
    foreach(((i, W),) -> println(io, " W[$i] = ", W), enumerate(H.data))
    return nothing
end

TensorKit.spacetype(::MPOHamiltonian{T}) where {T} = spacetype(T)
function TensorKit.storagetype(::Union{MPOHamiltonian{T},Type{MPOHamiltonian{T}}}) where {T}
    return TensorKit.storagetype(T)
end

virtualdim(H::MPOHamiltonian) = length(left_virtualspace(H, 1))
period(H::MPOHamiltonian) = length(H.data)

#default constructor
MPOHamiltonian(x::AbstractArray{<:Any,3}) = MPOHamiltonian(SparseMPO(x).data)

#allow passing in regular tensormaps
# MPOHamiltonian(t::TensorMap) = MPOHamiltonian(decompose_localmpo(add_util_leg(t)));

#a very simple utility constructor; given our "localmpo", constructs a mpohamiltonian
function MPOHamiltonian(x::Array{T,1}) where {T<:MPOTensor{Sp}} where {Sp}
    nOs = PeriodicArray{Union{scalartype(T),T}}(
        fill(zero(scalartype(T)), 1, length(x) + 1, length(x) + 1)
    )

    for (i, t) in enumerate(x)
        nOs[1, i, i + 1] = t
    end

    nOs[1, 1, 1] = one(scalartype(T))
    nOs[1, end, end] = one(scalartype(T))

    return MPOHamiltonian(SparseMPO(nOs))
end

left_virtualspace(H::MPOHamiltonian, i::Int) = space(H[i], 1)
right_virtualspace(H::MPOHamiltonian, i::Int) = space(H[i], 4)

left_virtualdim(H::MPOHamiltonian, i::Int) = size(H[i], 1)
right_virtualdim(H::MPOHamiltonian, i::Int) = size(H[i], 4)

# function Base.getproperty(h::MPOHamiltonian, f::Symbol)
#     if f in (:odim, :period, :imspaces, :domspaces, :Os, :pspaces)
#         return getproperty(h.data, f)
#     else
#         return getfield(h, f)
#     end
# end

Base.getindex(x::MPOHamiltonian, a) = x.data[a];

Base.eltype(x::MPOHamiltonian) = eltype(x.data)
VectorInterface.scalartype(::Type{MPOHamiltonian{T}}) where {T} = scalartype(T)
Base.size(x::MPOHamiltonian) = (x.period, x.odim, x.odim)
Base.size(x::MPOHamiltonian, i) = size(x)[i]
Base.length(x::MPOHamiltonian) = length(x.data)
TensorKit.space(x::MPOHamiltonian, i) = space(x.data, i)
Base.copy(x::MPOHamiltonian) = MPOHamiltonian(copy(x.data))
Base.iterate(x::Union{MPOHamiltonian,InfiniteMPO}, args...) = iterate(x.data, args...)
"
checks if ham[:,i,i] = 1 for every i
"
function Base.isone(H::MPOHamiltonian, i::Int)
    I = CartesianIndex(i, 1, 1, i)
    for i in 1:(period(H))
        if !(haskey(H[i], I) && H[i][I] isa TensorKit.BraidingTensor)
            return false
        end
    end
    return true
end

function Base.iszero(H::MPOHamiltonian, i::Int)
    I = CartesianIndex(i, 1, 1, i)
    for h in H.data
        I in keys(parent(h)) && return false
    end
    return true
end

"
to be valid in the thermodynamic limit, these hamiltonians need to have a peculiar structure
"
function sanitycheck(ham::MPOHamiltonian)
    for i in 1:period(ham)
        ham[i][1, 1, 1, 1] isa TensorKit.BraidingTensor || return false
        @assert ham[i][end, 1, 1, end] isa TensorKit.BraidingTensor || return false
        
        for j in 1:(left_virtualdim(ham, i)), k in 1:(j - 1)
            CartesianIndex(j, 1, 1, k) ∈ keys(ham[i]) || return false
        end
    end
    return true
end

#addition / substraction
Base.:-(H::MPOHamiltonian) = -one(scalartype(H)) * H
function Base.:+(H::MPOHamiltonian, λs::AbstractVector{<:Number})
    length(λs) == period(H) ||
        throw(ArgumentError("periodicity should match $(period(H)) ≠ $(length(λs))"))
    H′ = copy(H)
    
    foreach(H′.data, λs) do h, λ
        D = h[1, 1, 1, end]
        h[1, 1, 1, end] = add!(D, isomorphism(storagetype(D), space(D)), λ)
    end
    return H′
end

Base.:-(e::AbstractVector, a::MPOHamiltonian) = -1.0 * a + e
Base.:+(e::AbstractVector, a::MPOHamiltonian) = a + e
Base.:-(a::MPOHamiltonian, e::AbstractVector) = a + (-e)

# Base.:+(a::H1, b::H2) where {H1<:MPOHamiltonian,H2<:MPOHamiltonian} = +(promote(a, b)...)
function Base.:+(a::MPOHamiltonian{T}, b::MPOHamiltonian{T}) where {T}
    period(a) == period(b) ||
        throw(ArgumentError("periodicity should match $(period(a)) ≠ $(period(b))"))
    
    @assert sanitycheck(a) "a is not a valid hamiltonian"
    @assert sanitycheck(b) "b is not a valid hamiltonian"

    Hnew = map(a.data, b.data) do h1, h2
        Vₗ₁ = left_virtualspace(h1)
        Vₗ₂ = left_virtualspace(h2)
        @assert Vₗ₁[1] == Vₗ₂[1] && Vₗ₁[end] == Vₗ₂[end] "trivial spaces should match"
        Vₗ = Vₗ₁[1:end-1] ⊕ Vₗ₂[2:end]
        
        Vᵣ₁ = right_virtualspace(h1)
        Vᵣ₂ = right_virtualspace(h2)
        @assert Vᵣ₁[1] == Vᵣ₂[1] && Vᵣ₁[end] == Vᵣ₂[end] "trivial spaces should match"
        Vᵣ = Vᵣ₁[1:end-1] ⊕ Vᵣ₂[2:end]
        
        Wnew = T(undef, Vₗ ⊗ space(h1, 2), space(h1, 3)' ⊗ Vᵣ')
        
        # add blocks from first hamiltonian
        for (I, O) in nonzero_pairs(h1)
            if I[1] == 1
                if I[4] == 1
                    # 1 block
                    Wnew[I] = O
                elseif I[4] == size(h1, 4)
                    # D block
                    Wnew[1, 1, 1, end] = O
                else
                    # C block
                    Wnew[I] = O
                end
            elseif I[4] == size(h1, 4)
                if I[1] == size(h1, 1)
                    # 1 block
                    Wnew[1, 1, 1, end] = O
                else
                    # B block
                    Wnew[I[1], 1, 1, end] = O
                end
            else
                # A block
                Wnew[I] = O
            end
        end
        
        # add blocks from second hamiltonian
        for (I, O) in nonzero_pairs(h2)
            if I[1] == 1
                if I[4] == 1
                    # 1 block - already done
                elseif I[4] == size(h2, 4)
                    # D block
                    Wnew[1, 1, 1, end] += O
                else
                    # C block
                    shift = CartesianIndex(0, 0, 0, size(h1, 4) - 2)
                    Wnew[I + shift] = O
                end
            elseif I[4] == size(h2, 4)
                if I[1] == size(h1, 1)
                    # 1 block - already done
                else
                    # B block
                    shift = CartesianIndex(size(h1, 1) - 2, 0, 0, size(h1, 4) - 2)
                    Wnew[I + shift] = O
                end
            else
                # A block
                shift = CartesianIndex(size(h1, 1) - 2, 0, 0, size(h1, 4) - 2)
                Wnew[I + shift] = O
            end
        end
        return Wnew
    end
    
    return MPOHamiltonian(Hnew)
end

Base.:-(a::MPOHamiltonian, b::MPOHamiltonian) = a + (-b)

#multiplication
Base.:*(λ::Number, H::MPOHamiltonian) = H * λ
function Base.:*(H::MPOHamiltonian, λ::Number)
    Hλ = copy(H)
    foreach(Hλ.data) do h
        # multiply scalar with start of every interaction
        # this avoids double counting
        rmul!(h[1, 1, 1, :], λ)
    end
    return Hλ
end

function Base.:*(a::H1, b::H2) where {H1<:MPOHamiltonian,H2<:MPOHamiltonian}
    a′, b′ = promote(a, b)
    return *(a′, b′)
end
function Base.:*(b::H, a::H) where {H<:MPOHamiltonian}
    T = eltype(b.data)
    S = spacetype(T)
    period(b) == period(a) || throw(ArgumentError("periodicity should match: $(period(b)) ≠ $(period(a))"))
    
    E = promote_type(scalartype(b), scalartype(a))
    Fs = PeriodicArray(fuser.(E, left_virtualspace.(parent(a)), left_virtualspace.(parent(b))))
    
    C = similar(b.data)
    for i in 1:period(b)
        C[i] = T(undef, space(Fs[i], 1) ⊗ physicalspace(b, i) ← physicalspace(a, i) ⊗ space(Fs[i + 1], 1))
        @plansor C[i][-1 -2; -3 -4] = Fs[i][-1; 1 2] * a[i][1 5; -3 3] * b[i][2 -2; 5 4] * conj(Fs[i + 1][-4; 3 4])
        
        # restore sparsity -> when both factors are braidingtensors, we know that the
        # result can again be represented as a braidingtensor
        cinds = CartesianIndices((size(a[i], 1), size(b[i], 1)))
        for j in axes(C[i], 1), k in axes(C[i], 4)
            rowinds = cinds[j]
            colinds = cinds[k]
            Ia = CartesianIndex(rowinds[1], 1, 1, colinds[1])
            Ib = CartesianIndex(rowinds[2], 1, 1, colinds[2])
            if (haskey(a[i], Ia) && a[i][Ia] isa TensorKit.BraidingTensor) &&
                (haskey(b[i], Ib) && b[i][Ib] isa TensorKit.BraidingTensor)
                V = getsubspace(space(C[i]), CartesianIndex(j, 1, 1, k))
                C[i][j, 1, 1, k] = TensorKit.BraidingTensor{S,Matrix{E}}(V[2], V[1])
            end
        end
    end
    return MPOHamiltonian(C)
end

function Base.:(^)(a::MPOHamiltonian, n::Int)
    n >= 1 || throw(DomainError(n, "n should be a positive integer"))
    return Base.power_by_squaring(a, n)
end

Base.repeat(x::MPOHamiltonian, n::Int) = MPOHamiltonian(repeat(x.data, n))
function Base.conj(a::MPOHamiltonian)
    return MPOHamiltonian(map(a.data) do x
        @plansor x′[-1 -2; -3 -4] := conj(x[-1 -3; -2 -4])
    end)
end

Base.lastindex(h::MPOHamiltonian) = lastindex(h.data);

Base.convert(::Type{DenseMPO}, H::MPOHamiltonian) = convert(DenseMPO, convert(SparseMPO, H))
Base.convert(::Type{SparseMPO}, H::MPOHamiltonian{T}) where {T} = InfiniteMPO{T}(H.data)

Base.:*(H::MPOHamiltonian, mps::InfiniteMPS) = convert(DenseMPO, H) * mps

function add_physical_charge(O::MPOHamiltonian, charges::AbstractVector)
    return MPOHamiltonian(add_physical_charge(O.data, charges))
end

# promotion and conversion
# ------------------------
function Base.promote_rule(
    ::Type{<:MPOHamiltonian{<:BlockTensorMap{S,N₁,N₂,T₁,N}}}, ::Type{<:MPOHamiltonian{<:BlockTensorMap{S,N₁,N₂,T₂,N}}}
) where {S,N₁,N₂,T₁,T₂,N}
    T = promote_type(T₁, T₂)
    return MPOHamiltonian{BlockTensorMap{S,N₁,N₂,T,N}}
end

function Base.convert(::Type{MPOHamiltonian{T}}, x::MPOHamiltonian) where {T}
    typeof(x) == MPOHamiltonian{T} && return x
    return MPOHamiltonian{T}(convert.(T, x.data))
end
