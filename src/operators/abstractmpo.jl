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

function ismpoidentity(O::MPOTensor{S}; tol=eps(real(scalartype(O)))^3 / 4) where {S}
    O isa BraidingTensor && return true
    τ = TensorKit.BraidingTensor{S,storagetype(O)}(space(O, 2), space(O, 1))
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


TensorKit.spacetype(::Union{O,Type{O}}) where {O<:AbstractMPO} = spacetype(tensortype(O))
TensorKit.sectortype(::Union{O,Type{O}}) where {O<:AbstractMPO} = sectortype(tensortype(O))
function TensorKit.storagetype(::Union{O,Type{O}}) where {O<:AbstractMPO}
    return storagetype(tensortype(O))
end

VectorInterface.scalartype(::Type{O}) where {O<:AbstractMPO} = scalartype(eltype(O))

Base.eltype(O::AbstractMPO) = eltype(parent(O))
Base.length(O::AbstractMPO) = length(parent(O))

Base.iterate(O::AbstractMPO, args...) = iterate(parent(O), args...)

Base.getindex(O::AbstractMPO, i::Int) = getindex(parent(O), i)
Base.setindex!(O::AbstractMPO, v, i::Int) = setindex!(parent(O), v, i)
Base.lastindex(O::AbstractMPO) = lastindex(parent(O))
Base.checkbounds(::Type{Bool}, O::AbstractMPO, args...) = checkbounds(parent(O), args...)

Base.similar(O::MPO, args...) where {MPO<:AbstractMPO} = MPO(similar(parent(O), args...))
Base.copy(O::MPO) where {MPO<:AbstractMPO} = MPO(copy(parent(O)))
Base.deepcopy(O::MPO) where {MPO<:AbstractMPO} = MPO(deepcopy(parent(O)))
Base.repeat(O::MPO, args...) where {MPO<:AbstractMPO} = MPO(repeat(parent(O), args...))
