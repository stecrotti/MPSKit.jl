# MPO Tensor types
# ----------------

const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S}
const SparseMPOTensor{T<:MPOTensor} = BlockTensorMap{S,2,2,T,4} where {S}
const AbstractMPOTensor{S} = Union{SparseMPOTensor{<:MPOTensor{S}},MPOTensor{S}} where {S}

left_virtualspace(O::AbstractMPOTensor) = space(O, 1)
right_virtualspace(O::AbstractMPOTensor) = space(O, 4)
physicalspace(O::AbstractMPOTensor) = space(O, 2)

left_virtualsize(O::MPOTensor) = 1
left_virtualsize(O::SparseMPOTensor) = size(O, 1)
right_virtualsize(O::MPOTensor) = 1
right_virtualsize(O::SparseMPOTensor) = size(O, 4)

function ismpoidentity(O::MPOTensor; tol=eps(scalartype(O))^3 / 4)
    O isa BraidingTensor && return true
    τ = TensorKit.BraidingTensor{S,storagetype(O)}(space(O, 1), space(O, 2))
    return isapprox(O, τ; atol=tol)
end

# MPO types
# ---------

abstract type AbstractMPO end
abstract type AbstractFiniteMPO <: AbstractMPO end
abstract type AbstractInfiniteMPO <: AbstractMPO end

left_virtualspace(O::AbstractMPO, i::Int) = left_virtualspace(O[i])
right_virtualspace(O::AbstractMPO, i::Int) = right_virtualspace(O[i])
physicalspace(O::AbstractMPO, i::Int) = physicalspace(O[i])
physicalspace(H::AbstractMPO) = ProductSpace(ntuple(i -> physicalspace(H[i]), length(H)))

TensorKit.space(O::AbstractMPO, i::Int) = physicalspace(O, i)

left_virtualsize(H::AbstractMPO, i::Int) = left_virtualsize(H[i])
right_virtualsize(H::AbstractMPO, i::Int) = right_virtualsize(H[i])

"""
    tensortype(H::Union{O,Type{O}}) where {O<:AbstractMPO}

Return the type of tensors of a Matrix Product Operator.
"""
function tensortype end
tensortype(H::AbstractMPO) = tensortype(typeof(H))

TensorKit.spacetype(::Union{O,Type{O}}) where {O<:AbstractMPO} = spacetype(tensortype(O))
TensorKit.sectortype(::Union{O,Type{O}}) where {O<:AbstractMPO} = sectortype(tensortype(O))
function TensorKit.storagetype(::Union{O,Type{O}}) where {O<:AbstractMPO}
    return storagetype(tensortype(O))
end

VectorInterface.scalartype(::Type{O}) where {O<:AbstractMPO} = scalartype(tensortype(O))

Base.eltype(O::AbstractMPO) = eltype(parent(O))
Base.length(a::AbstractMPO) = length(parent(a))

Base.iterate(x::AbstractMPO, args...) = iterate(parent(x), args...)

Base.getindex(a::AbstractMPO, i::Int) = getindex(parent(a), i)
Base.setindex!(a::AbstractMPO, v, i::Int) = setindex!(parent(a), v, i)
Base.checkbounds(::Type{Bool}, O::AbstractMPO, args...) = checkbounds(parent(O), args...)

Base.similar(x::MPO, args...) where {MPO<:AbstractMPO} = MPO(similar(parent(x), args...))
Base.copy(x::MPO) where {MPO<:AbstractMPO} = MPO(copy(parent(x)))
Base.deepcopy(x::MPO) where {MPO<:AbstractMPO} = MPO(deepcopy(parent(x)))
Base.repeat(x::MPO, args...) where {MPO<:AbstractMPO} = MPO(repeat(parent(x), args...))
