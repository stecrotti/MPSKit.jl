"
    SparseMPO - used to represent both time evolution mpos and hamiltonians
"

struct InfiniteMPO{T<:AbstractMPOTensor} <: AbstractInfiniteMPO
    data::PeriodicVector{T}

    # regular constructor
    function InfiniteMPO{T}(data::PeriodicVector{T}) where {T<:AbstractMPOTensor}
        return new{T}(data)
    end

    # constructor with guaranteed space checks
    function InfiniteMPO(data::PeriodicVector{T}) where {T<:AbstractMPOTensor}
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

const SparseMPO{T<:SparseMPOTensor} = InfiniteMPO{T}
const DenseMPO{T<:MPOTensor} = InfiniteMPO{T}

# Constructors
# ------------

function InfiniteMPO(data::AbstractVector{<:AbstractMPOTensor})
    return InfiniteMPO(PeriodicArray(data))
end

function InfiniteMPO(data::AbstractArray{Union{T,E},3}) where {T<:MPOTensor,E<:Number}
    @assert scalartype(T) == E "scalar type should match mpo scalartype"

    L, left_virtualsz, right_virtualsz = size(data)
    @assert left_virtualsz == right_virtualsz "MPOs should be square"

    # deduce spaces from tensors
    S = spacetype(T)
    virtualspaces = PeriodicArray([Vector{Union{Missing,S}}(missing, left_virtualsz)
                                   for _ in 1:L])
    physicalspaces = Vector{Union{Missing,S}}(missing, L)

    isused = trues(size(data))
    ischanged = true
    while ischanged
        ischanged = false

        for I in CartesianIndices(data)
            isused[I] && continue # skip information that is already known
            i, j, k = I.I
            if data[I] isa T
                P = physicalspace(data[I])
                Vₗ = left_virtualspace(data[I])
                Vᵣ = right_virtualspace(data[I])

                if ismissing(physicalspaces[i])
                    physicalspaces[i] = P
                else
                    P == physicalspaces[i] ||
                        throw(ArgumentError("physical space mismatch at $(I.I)"))
                end

                if ismissing(virtualspaces[i][j])
                    virtualspaces[i][j] = Vₗ
                else
                    Vₗ == virtualspaces[i][j] ||
                        throw(ArgumentError("left virtual space mismatch at $(I.I)"))
                end

                if ismissing(virtualspaces[i + 1][k])
                    virtualspaces[i + 1][k] = Vᵣ
                else
                    Vᵣ == virtualspaces[i + 1][k] ||
                        throw(ArgumentError("right virtual space mismatch at $(I.I)"))
                end
                isused[I] = true

                # not necessarily changed, but tensors are only checked first time around
                # so something should definitely have changed
                ischanged = true

            elseif !iszero(data[I])
                ismissing(virtualspaces[i][j]) && ismissing(virtualspaces[i + 1][k]) &&
                    continue

                if ismissing(virtualspaces[i][j])
                    # left space can be deduced from right
                    Vᵣ = virtualspaces[i + 1][k]
                    Vₗ = Vᵣ
                    virtualspaces[i][j] = Vₗ
                    ischanged = true
                elseif ismissing(virtualspaces[i + 1][k])
                    # right space can be deduced from left
                    Vₗ = virtualspaces[i][j]
                    Vᵣ = Vₗ
                    virtualspaces[i + 1][k] = Vᵣ
                    ischanged = true
                else
                    # both spaces are assigned
                    Vₗ = virtualspaces[i][j]
                    Vᵣ = virtualspaces[i + 1][k]
                    Vₗ == Vᵣ ||
                        throw(ArgumentError("virtual space mismatch at $(I.I)"))
                end

                # check that braiding is possible
                _can_unambiguously_braid(Vₗ) ||
                    throw(ArgumentError("Ambiguous braiding operator at $(I.I)"))

                isused[I] = true
            end
        end
    end

    # check that all spaces are assigned
    for i in 1:L
        ismissing(physicalspaces[i]) &&
            throw(ArgumentError("Physical space at $i is not assigned"))
        replace!(virtualspaces[i], missing => oneunit(S)) # this should not cause problems?
    end

    # construct blocktensors
    τtype = TensorKit.BraidingTensor{S,TensorKit.storagetype(M)}
    ttype = Union{T,τtype}

    Ws = map(1:L) do i
        Vₗ = SumSpace(virtualspaces[i]...)
        Vᵣ = SumSpace(virtualspaces[i + 1]...)
        P = SumSpace(physicalspaces[i])
        tdst = BlockTensorMap{S,2,2,ttype}(undef, Vₗ ⊗ P, P ⊗ Vᵣ)
        for j in axes(data, 2), k in axes(data, 3)
            if data[i, j, k] isa E
                iszero(data[i, j, k]) && continue
                τ = τtype(domain(BlockTensorKit.getsubspace(space(tdst), j, 1, 1, k))...)
                if isone(data[i, j, k])
                    tdst[j, 1, 1, k] = τ
                else
                    tdst[j, 1, 1, k] = scale!(τ, data[i, j, k])
                end
            else
                if ismpoidentity(data[i, j, k])
                    tdst[j, 1, 1, k] = τtype(domain(BlockTensorKit.getsubspace(space(tdst),
                                                                               j, 1, 1, k))...)
                else
                    tdst[j, 1, 1, k] = data[i, j, k]
                end
            end
        end
        return tdst
    end

    return InfiniteMPO(Ws)
end

# Attempt to deduce eltype information for non-strictly typed data
function InfiniteMPO(data::AbstractArray{<:Any,3})
    # deduce types
    tensortypes = Set()
    scalartypes = Set()
    for x in data
        if x isa AbstractTensorMap
            if numin(x) == numout(x) == 2
                push!(tensortypes, typeof(x))
            elseif numin(x) == numout(x) == 1
                push!(tensortypes, tensormaptype(spacetype(x), 2, 2, scalartype(x)))
            else
                throw(ArgumentError("$(typeof(x)) is not an MPO tensor or a single-site tensor"))
            end
        elseif x isa Number
            push!(scalartypes, typeof(x))
        else
            ismissing(x) ||
                throw(ArgumentError("data should only contain mpo tensors or scalars"))
        end
    end
    T = promote_type(tensortypes...)
    E = promote_type(scalartype(T), scalartypes...)

    # convert data
    data′ = similar(data, Union{E,T})
    for (i, x) in enumerate(data)
        if x isa AbstractTensorMap
            if numin(x) == numout(x) == 2
                data′[i] = convert(T, x)
            elseif numin(x) == numout(x) == 1
                data′[i] = convert(T, add_util_leg(x))
            else
                error("this should not happen")
            end
        elseif x isa Number
            data′[i] = convert(E, x)
        else
            data′[i] = zero(E)
        end
    end

    return InfiniteMPO(data′)
end

# promotion and conversion
# ------------------------

function Base.promote_rule(::Type{InfiniteMPO{T₁}}, ::Type{InfiniteMPO{T₂}}) where {T₁,T₂}
    return InfiniteMPO{promote_type(T₁, T₂)}
end

function Base.convert(::Type{InfiniteMPO{T₁}}, x::InfiniteMPO{T₂}) where {T₁,T₂}
    T₁ === T₂ && return x
    return InfiniteMPO{T₁}(map(Base.Fix1(convert, T₁), x.data))
end

function Base.convert(::Type{InfiniteMPO}, ψ::InfiniteMPS)
    return InfiniteMPO(map(ψ.AL) do al
                           @plansor tt[-1 -2; -3 -4] := al[-1 -2 1; 2] * τ[-3 2; -4 1]
                       end)
end

function Base.convert(::Type{InfiniteMPS}, O::InfiniteMPO)
    return InfiniteMPS(map(parent(O)) do o
                           @plansor A[-1 -2 -3; -4] := o[-1 -2; 1 2] * τ[1 2; -4 -3]
                       end)
end

# Linear Algebra
# --------------

function Base.:*(O::InfiniteMPO, ψ::InfiniteMPS)
    length(O) == length(ψ) ||
        throw(ArgumentError("Period mismatch: $(length(O)) ≠ $(length(ψ))"))
    Fs = PeriodicArray(fuser.(scalartype(ψ), left_virtualspace.(ψ.AL),
                              left_virtualspace.(parent(O))))
    AL = map(1:length(O)) do i
        @plansor t[-1 -2; -3] := ψ.AL[i][1 2; 3] * O[i][4 -2; 2 5] * Fs[i][-1; 1 4] *
                                 conj(Fs[i + 1][-3; 3 5])
    end
    return InfiniteMPS(AL)
end

Base.:(*)(O₁::InfiniteMPO, O₂::InfiniteMPO) = *(promote(O₁, O₂)...)
function Base.:*(O₁::H, O₂::H) where {T,H<:InfiniteMPO{T}}
    length(O₁) == length(O₂) ||
        throw(ArgumentError("Period mismatch: $(length(O₁)) ≠ $(length(O₂))"))

    S = spacetype(T)
    E = scalartype(T)
    Fs = PeriodicArray(fuser.(E, left_virtualspace.(parent(O₁)),
                              left_virtualspace.(parent(O₂))))

    O = map(1:length(O₁), parent(O₁), parent(O₂)) do i, o₁, o₂
        @plansor o[-1 -2; -3 -4] := Fs[i][-1; 1 2] * o₁[1 5; -3 3] * o₂[2 -2; 5 4] *
                                    conj(Fs[i + 1][-4; 3 4])
        if T <: SparseMPOTensor
            # restore sparsity -> when both factors are braidingtensors, we know that the
            # result can again be represented as a braidingtensor
            cinds = CartesianIndices((size(o₁, 1), size(o₂, 1)))
            for j in axes(o, 1), k in axes(o, 4)
                rowinds = cinds[j]
                colinds = cinds[k]
                Ia = CartesianIndex(rowinds[1], 1, 1, colinds[1])
                Ib = CartesianIndex(rowinds[2], 1, 1, colinds[2])
                if (haskey(o₁, Ia) && ismpoidentity(o₁[Ia])) &&
                   (haskey(o₂, Ib) && ismpoidentity(o₂[Ib]))
                    V = getsubspace(space(o), CartesianIndex(j, 1, 1, k))
                    o[j, 1, 1, k] = BraidingTensor{S,Matrix{E}}(V[2], V[1])
                end
            end
        end
        return o
    end

    return InfiniteMPO(O)
end

function Base.conj(O::InfiniteMPO)
    data′ = map(parent(O)) do o
        @plansor o′[-1 -2; -3 -4] := conj(o[-1 -3; -2 -4])
    end
    return InfiniteMPO(data′)
end

function TensorKit.dot(O₁::InfiniteMPO, O₂::InfiniteMPO)
    length(O₁) == length(O₂) ||
        throw(ArgumentError("Period mismatch: $(length(O₁)) ≠ $(length(O₂))"))
    return TensorKit.dot(convert.(InfiniteMPS, (O₁, O₂))...)
end

function TensorKit.dot(ψ₁::InfiniteMPS, O::InfiniteMPO, ψ₂::InfiniteMPS;
                       krylovdim=30)
    length(ψ₁) == length(O) == length(ψ₂) ||
        throw(ArgumentError("Period mismatch: $(length(ψ₁)) ≠ $(length(O)) ≠ $(length(ψ₂))"))

    T = TransferMatrix(ψ₂.AL, parent(O), ψ₁.AL)
    ρ₀ = similar(O[1], _firstspace(ψ₁.AL[1]) ⊗ _firstspace(O[1]) ← _firstspace(ψ₂.AL[1]))
    vals, _, info = eigsolve(T, ρ₀, 1, :LM, Arnoldi(; krylovdim=krylovdim))
    info.converged == 0 && @warn "dot mps not converged"
    return first(vals)
end

# Utility
# -------

"""
    remove_orphans(mpo::SparseMPO)

Prune all branches of the finite state machine that do not contribute. Additionally, attempt
to compact the representation as much as possible.
"""
function remove_orphans!(O::SparseMPO; tol=eps(real(scalartype(O)))^(3 / 4))
    # drop zeros
    for slice in parent(O)
        for (key, val) in nonzero_pairs(slice)
            norm(val) < tol && delete!(slice, key)
        end
    end

    # drop dead starts/ends
    changed = true
    while changed
        changed = false

        for i in 1:length(O)
            # slice empty columns on right or empty rows on left
            mask = filter(1:size(O[i], 4)) do j
                return j ∈ getindex.(nonzero_keys(O[i]), 1) ||
                       j ∈ getindex.(nonzero_keys(O[i + 1]), 4)
            end
            changed |= length(mask) == size(O[i], 4)
            O[i] = O[i][:, :, :, mask]
            O[i + 1] = O[i + 1][mask, :, :, :]
        end
    end

    return O
end

"""
    add_physical_charge(O::Union{SparseMPO{S},MPOHamiltonion{S}},
                        auxspaces::AbstractVector{I}) where {S,I}

create an operator which passes an auxiliary space.
"""
function add_physical_charge(O::InfiniteMPO, charges::AbstractVector{I}) where {I<:Sector}
    length(charges) == length(O) || throw(ArgumentError("unmatching lengths"))
    sectortype(S) == I ||
        throw(ArgumentError("Unmatching chargetypes $sectortype(S) and $I"))

    auxspaces = map(c -> Vect[I](c => 1), charges)
    O_new = copy(O)
    O_new.pspaces .= fuse.(O.pspaces, auxspaces)

    for i in eachindex(O.pspaces)
        F = unitary(O_new.pspaces[i] ← O.pspaces[i] ⊗ auxspaces[i])
        for (j, k) in MPSKit.opkeys(O_new[i])
            @plansor begin
                O_new[i][j, k][-1 -2; -3 -4] := F[-2; 1 2] * O[i][j, k][-1 1; 4 3] *
                                                τ[3 2; 5 -4] * conj(F[-3; 4 5])
            end
        end
    end

    return O_new
end

Base.isone(x::SparseMPOTensor, I::Vararg{Int,4}) = isone(x.scalars[I...])
function Base.iszero(x::SparseMPOTensor, I::Vararg{Int,4})
    return iszero(x.scalars[I...]) && !haskey(x.tensors.data.data, CartesianIndex(I))
end
