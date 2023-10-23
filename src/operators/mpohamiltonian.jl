"
    MPOHamiltonian

    represents a general periodic quantum hamiltonian

    really just a sparsempo, with some garantuees on its structure
"


struct MPOHamiltonian{T<:SparseMPOTensor} <: AbstractMPO
    data::PeriodicVector{T}
    function MPOHamiltonian(data::AbstractVector{T}) where {T<:SparseMPOTensor}
        for i in eachindex(data)
            Vₗ = left_virtualspace(data[i])
            Vᵣ = dual(right_virtualspace(data[i - 1]))
            Vₗ == Vᵣ ||
                throw(SpaceMismatch("Incompatible virtual spaces at $i:\n$Vₗ ≠ $Vᵣ"))
            space(data[i], 2) == dual(space(data[i], 3)) ||
                throw(TensorKit.SpaceMismatch("Incompatible physical spaces at $i"))
        end
        return new{T}(convert(PeriodicArray, data))
    end
end

# BlockTensorKit.blocktype(::MPOHamiltonian{T}) where {T} = blocktype(T)

physicalspace(H::MPOHamiltonian, i::Int) = physicalspace(H[i])

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
    
    return MPOHamiltonian(PeriodicVector([W]))
end

Base.parent(H::MPOHamiltonian) = H.data

function Base.show(io::IO, ::MIME"text/plain", H::Union{MPOHamiltonian,SparseMPO})
    typestr = H isa MPOHamiltonian ? "MPOHamiltonian" : "SparseMPO"
    println(io, "$(period(H))-periodic $(typestr){$(spacetype(H))}:")
    for (i, W) in enumerate(H)
        println(io, " W[$i] =")
        W′ = reshape(parent(W), size(W, 1), size(W, 4))
        if max(size(W′)...) <= 16
            _print_jordanmpo(io, W′)
        else
            _print_braille(io, W′)
        end
        
    end
end

function _print_jordanmpo(io, W)
    print_str = fill('.', size(W))
    for (j, O) in nonzero_pairs(W)
        if O isa TensorKit.BraidingTensor
            print_str[j] = 'τ'
        elseif j[1] == 1 && j[2] == size(W, 2)
            print_str[j] = 'D'
        elseif j[1] == 1
            print_str[j] = 'C'
        elseif j[2] == size(W, 2)
            print_str[j] = 'B'
        else
            print_str[j] = 'A'
        end
    end
    for (j, row) in enumerate(eachrow(reshape(print_str, size(W, 1), size(W, 2))))
        start = j == 1 ? "┌" : j == size(W, 1) ? "└" : "│"
        stop = j == 1 ? "┐" : j == size(W, 1) ? "┘" : "│"
        println(io, start, join(row, ' '), stop)
    end
end

# adapted from SparseArrays.jl
const brailleBlocks = UInt16['⠁', '⠂', '⠄', '⡀', '⠈', '⠐', '⠠', '⢀']
function _print_braille(io, W)
    m, n = size(W)
    
    # The maximal number of characters we allow to display the matrix
    local maxHeight::Int, maxWidth::Int
    maxHeight = displaysize(io)[1] - 4 # -4 from [Prompt, header, newline after elements, new prompt]
    maxWidth = displaysize(io)[2] ÷ 2
    
    if get(io, :limit, true) && (m > 4maxHeight || n > 2maxWidth)
        s = min(2maxWidth / n, 4maxHeight / m)
        scaleHeight = floor(Int, s * m)
        scaleWidth = floor(Int, s * n)
    else
        scaleHeight = m
        scaleWidth = n
    end
    
    # Make sure that the matrix size is big enough to be able to display all
    # the corner border characters
    if scaleHeight < 8
        scaleHeight = 8
    end
    if scaleWidth < 4
        scaleWidth = 4
    end
    
    brailleGrid = fill(UInt16(10240), (scaleWidth - 1) ÷ 2 + 4, (scaleHeight - 1) ÷ 4 + 1)
    brailleGrid[1,:] .= '⎢'
    brailleGrid[end-1,:] .= '⎥'
    brailleGrid[1,1] = '⎡'
    brailleGrid[1,end] = '⎣'
    brailleGrid[end-1,1] = '⎤'
    brailleGrid[end-1,end] = '⎦'
    brailleGrid[end, :] .= '\n'

    rowscale = max(1, scaleHeight - 1) / max(1, m - 1)
    colscale = max(1, scaleWidth - 1) / max(1, n - 1)
    
    for I in nonzero_keys(W)
        si = round(Int, (I[1] - 1) * rowscale + 1)
        sj = round(Int, (I[2] - 1) * colscale + 1)
        
        k = (sj - 1) ÷ 2 + 2
        l = (si - 1) ÷ 4 + 1
        p = ((sj - 1) % 2) * 4 + ((si - 1) % 4 + 1)

        brailleGrid[k, l] |= brailleBlocks[p]
    end
    
    foreach(c -> print(io, Char(c)), @view brailleGrid[1:end-1])
    return nothing
end

TensorKit.spacetype(::MPOHamiltonian{T}) where {T} = spacetype(T)
function TensorKit.storagetype(::Union{MPOHamiltonian{T},Type{MPOHamiltonian{T}}}) where {T}
    return TensorKit.storagetype(T)
end

virtualdim(H::MPOHamiltonian) = length(left_virtualspace(H, 1))
period(H::MPOHamiltonian) = length(H.data)

#default constructor
MPOHamiltonian(x::AbstractArray{<:Any,3}) = MPOHamiltonian(SparseMPO(x))

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

function Base.getproperty(h::MPOHamiltonian, f::Symbol)
    if f in (:odim, :period, :imspaces, :domspaces, :Os, :pspaces)
        return getproperty(h.data, f)
    else
        return getfield(h, f)
    end
end

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
    for i in 1:(ham.period)
        @assert isid(ham[i][1, 1])[1]
        @assert isid(ham[i][ham.odim, ham.odim])[1]

        for j in 1:(ham.odim), k in 1:(j - 1)
            contains(ham[i], j, k) && return false
        end
    end

    return true
end

#addition / substraction
Base.:-(a::MPOHamiltonian) = -one(scalartype(a)) * a
function Base.:+(a::MPOHamiltonian, e::AbstractVector)
    length(e) == a.period ||
        throw(ArgumentError("periodicity should match $(a.period) ≠ $(length(e))"))

    nOs = copy(a.data) # we don't want our addition to change different copies of the original hamiltonian

    for c in 1:(a.period)
        nOs[c][1, end] +=
            e[c] * isomorphism(
                storagetype(nOs[c][1, end]),
                codomain(nOs[c][1, end]),
                domain(nOs[c][1, end]),
            )
    end

    return MPOHamiltonian(nOs)
end
Base.:-(e::AbstractVector, a::MPOHamiltonian) = -1.0 * a + e
Base.:+(e::AbstractVector, a::MPOHamiltonian) = a + e
Base.:-(a::MPOHamiltonian, e::AbstractVector) = a + (-e)

Base.:+(a::H1, b::H2) where {H1<:MPOHamiltonian,H2<:MPOHamiltonian} = +(promote(a, b)...)
function Base.:+(a::H, b::H) where {H<:MPOHamiltonian}
    # this is a bit of a hack because I can't figure out how to make this more specialised
    # than the fallback which promotes, while still having access to S,T, and E.
    S, T, E = H.parameters

    a.period == b.period ||
        throw(ArgumentError("periodicity should match $(a.period) ≠ $(b.period)"))
    @assert sanitycheck(a)
    @assert sanitycheck(b)

    nodim = a.odim + b.odim - 2
    nOs = PeriodicArray{Union{E,T},3}(fill(zero(E), a.period, nodim, nodim))

    for pos in 1:(a.period)
        for (i, j) in keys(a[pos])
            #A block
            if (i < a.odim && j < a.odim)
                nOs[pos, i, j] = a[pos][i, j]
            end

            #right side
            if (i < a.odim && j == a.odim)
                nOs[pos, i, nodim] = a[pos][i, j]
            end
        end

        for (i, j) in keys(b[pos])

            #upper Bs
            if (i == 1 && j > 1)
                if nOs[pos, 1, a.odim + j - 2] isa T
                    nOs[pos, 1, a.odim + j - 2] += b[pos][i, j]
                else
                    nOs[pos, 1, a.odim + j - 2] = b[pos][i, j]
                end
            end

            #B block
            if (i > 1 && j > 1)
                nOs[pos, a.odim + i - 2, a.odim + j - 2] = b[pos][i, j]
            end
        end
    end

    return MPOHamiltonian{S,T,E}(SparseMPO(nOs))
end
Base.:-(a::MPOHamiltonian, b::MPOHamiltonian) = a + (-b)

#multiplication
Base.:*(b::Number, a::MPOHamiltonian) = a * b
function Base.:*(a::MPOHamiltonian, b::Number)
    nOs = copy(a.data)

    for i in 1:(a.period), j in 1:(a.odim - 1)
        nOs[i][j, a.odim] *= b
    end
    return MPOHamiltonian(nOs)
end

function Base.:*(b::MPOHamiltonian{T}, a::MPOHamiltonian{T}) where {T}
    S = spacetype(T)
    period(b) == period(a) || throw(ArgumentError("periodicity should match: $(period(b)) ≠ $(period(a))"))
    
    E = promote_type(scalartype(b), scalartype(a))
    Fs = PeriodicArray(fuser.(E, left_virtualspace.(parent(a)), left_virtualspace.(parent(b))))
    
    C = similar(b.data)
    for i in 1:period(b)
        C[i] = BlockTensorMap(undef_blocks, blocktype(b), space(Fs[i], 1) ⊗ physicalspace(b, i) ← physicalspace(a, i) ⊗ space(Fs[i + 1], 1))
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
    ::Type{MPOHamiltonian{T₁}}, ::Type{MPOHamiltonian{T₂}}
) where {T₁,T₂}
    return MPOHamiltonian{promote_type(T₁, T₂)}
end

function Base.convert(::Type{MPOHamiltonian{T}}, x::MPOHamiltonian) where {T}
    typeof(x) == MPOHamiltonian{T} && return x
    return MPOHamiltonian{T}(convert.(T, x.data))
end
