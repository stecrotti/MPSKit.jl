abstract type AbstractMPO end

const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S}
const SparseMPOTensor{T<:MPOTensor} = BlockTensorMap{S,2,2,T,4} where {S}

const AbstractMPOTensor{S} = Union{SparseMPOTensor{<:MPOTensor{S}}, MPOTensor{S}} where {S}

left_virtualspace(O::AbstractMPOTensor) = space(O, 1)
left_virtualspace(O::BlockTensorMap{S,2,2}) where {S} = space(O, 1)
right_virtualspace(O::AbstractMPOTensor) = space(O, 4)
physicalspace(O::AbstractMPOTensor) = space(O, 2)
