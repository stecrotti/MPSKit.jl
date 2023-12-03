# this object represents a sparse mpo at a single position
"""
    SparseMPOSlice{S,T,E} <: AbstractArray{T,2}

A view of a sparse MPO at a single position.

# Fields
- `Os::AbstractMatrix{Union{T,E}}`: matrix of operators.
- `domspaces::AbstractVector{S}`: list of left virtual spaces.
- `imspaces::AbstractVector{S}`: list of right virtual spaces.
- `pspace::S`: physical space.
"""
struct SparseMPOSlice{S,T,E} <: AbstractArray{T,2}
    Os::SubArray{Union{T,E},
                 2,
                 PeriodicArray{Union{T,E},3},
                 Tuple{Int,Base.Slice{Base.OneTo{Int}},Base.Slice{Base.OneTo{Int}}},
                 false}
    domspaces::SubArray{S,1,PeriodicArray{S,2},Tuple{Int,Base.Slice{Base.OneTo{Int}}},false}
    imspaces::SubArray{S,1,PeriodicArray{S,2},Tuple{Int,Base.Slice{Base.OneTo{Int}}},false}
    pspace::S
end

function Base.getproperty(x::SparseMPOSlice, s::Symbol)
    if s == :odim
        return size(x, 1)
    else
        return getfield(x, s)
    end
end

#methods it must extend to be an abstractarray
Base.size(sl::SparseMPOSlice) = size(sl.Os)

function Base.getindex(x::SparseMPOSlice{S,T,E}, a::Int, b::Int)::T where {S,T,E}
    a <= x.odim && b <= x.odim || throw(BoundsError(x, [a, b]))
    if x.Os[a, b] isa E
        if x.Os[a, b] == zero(E)
            return fill_data!(TensorMap(x -> storagetype(T)(undef, x),
                                        x.domspaces[a] * x.pspace,
                                        x.pspace * x.imspaces[b]'),
                              zero)
        else
            return @plansor temp[-1 -2; -3 -4] := (x.Os[a, b] * isomorphism(storagetype(T),
                                                                            x.domspaces[a] * x.pspace,
                                                                            x.imspaces[b]' * x.pspace))[-1 -2
                                                                                                        1 2] *
                                                  τ[1 2; -3 -4]
        end
    else
        return x.Os[a, b]
    end
end

function Base.setindex!(x::SparseMPOSlice{S,T,E}, v::T, a::Int, b::Int) where {S,T,E}
    a <= x.odim && b <= x.odim || throw(BoundsError(x, [a, b]))
    (ii, scal) = isid(v)

    if ii
        x.Os[a, b] = scal ≈ one(scal) ? one(scal) : scal
    elseif v ≈ zero(v)
        x.Os[a, b] = zero(E)
    else
        x.Os[a, b] = v
    end

    return x
end

#utility methods
function Base.keys(x::SparseMPOSlice)
    return Iterators.filter(a -> contains(x, a[1], a[2]), product(1:(x.odim), 1:(x.odim)))
end
function Base.keys(x::SparseMPOSlice, ::Colon, t::Int)
    return Iterators.filter(a -> contains(x, a, t), 1:(x.odim))
end
function Base.keys(x::SparseMPOSlice, t::Int, ::Colon)
    return Iterators.filter(a -> contains(x, t, a), 1:(x.odim))
end

opkeys(x::SparseMPOSlice) = Iterators.filter(a -> !isscal(x, a[1], a[2]), keys(x));
scalkeys(x::SparseMPOSlice) = Iterators.filter(a -> isscal(x, a[1], a[2]), keys(x));

function opkeys(x::SparseMPOSlice, ::Colon, a::Int)
    return Iterators.filter(t -> contains(x, t, a) && !isscal(x, t, a), 1:(x.odim))
end;
function opkeys(x::SparseMPOSlice, a::Int, ::Colon)
    return Iterators.filter(t -> contains(x, a, t) && !isscal(x, a, t), 1:(x.odim))
end;

function scalkeys(x::SparseMPOSlice, ::Colon, a::Int)
    return Iterators.filter(t -> isscal(x, t, a), 1:(x.odim))
end;
function scalkeys(x::SparseMPOSlice, a::Int, ::Colon)
    return Iterators.filter(t -> isscal(x, a, t), 1:(x.odim))
end;

function Base.contains(x::SparseMPOSlice{S,T,E}, a::Int, b::Int) where {S,T,E}
    return !(x.Os[a, b] == zero(E))
end
function isscal(x::SparseMPOSlice{S,T,E}, a::Int, b::Int) where {S,T,E}
    return x.Os[a, b] isa E && contains(x, a, b)
end

# SparseMPOTensor
# ---------------
# struct SparseMPOTensor{S,T<:MPOTensor,E} <: MPOTensor{S}
#     tensors::BlockTensorMap{S,2,2,SparseArray{T,4}}
#     scalars::SparseArray{E,4}

#     function SparseMPOTensor(
#         tensors::BlockTensorMap{S,2,2,SparseArray{T,4}},
#         scalars::SparseArray{E,4}=SparseArray{scalartype(tensors),4}(
#             undef, size(tensors)
#         ),
#     ) where {S,T,E<:Number}
#         size(tensors) == size(scalars) ||
#             throw(ArgumentError("tensors and scalars should be equal in size"))
#         scalartype(tensors) === scalartype(scalars) ||
#             throw(ArgumentError("tensors and scalars should have equal scalartypes."))
#             # TODO: combine scalars with tensors where they exist
#             # TODO: check that scalars only appear where BraidingTensors can exist
#         return new{S,eltype(tensors),E}(tensors, scalars)
#     end
# end

# # AbstractArray Interface
# # -----------------------
# Base.size(t::SparseMPOTensor) = size(t.scalars)
# Base.ndims(::Type{<:SparseMPOTensor}) = 4
# Base.eltype(::Type{<:SparseMPOTensor{S,T,E}}) where {S,T,E} = Union{T,E}

# Base.getindex(t::SparseMPOTensor, I::Vararg{Int,4}) = getindex(t, CartesianIndex(I...))
# function Base.getindex(t::SparseMPOTensor, I::CartesianIndex{4})
#     if haskey(parent(t.tensors).data, I)
#         return t.tensors[I]
#     else
#         elspace = getsubspace(space(t), I)
#         @assert domain(elspace) == ProductSpace(reverse(codomain(elspace).spaces)) elspace
#         τ = TensorKit.BraidingTensor{spacetype(t),storagetype(eltype(t.tensors))}(elspace[2], elspace[1])
#         return scale(copy(τ), t.scalars[I])::eltype(t.tensors)
#     end
# end
# function Base.getindex(t::SparseMPOTensor, I::Vararg{Union{Int,AbstractVector{Int}}})
#     return SparseMPOTensor(
#         getindex(t.tensors, I...),
#         reshape(
#             getindex(t.scalars, I...), map(i -> i isa Colon ? size(t, i) : length(i), I)
#         ),
#     )
# end

# function Base.setindex!(t::SparseMPOTensor, v, I::Vararg{Int,4})
#     return setindex!(t, v, CartesianIndex(I))
# end
# function Base.setindex!(t::SparseMPOTensor, v, I::CartesianIndex{4})
#     if v isa Number
#         delete!(parent(t.tensors).data, I)
#         setindex!(t.scalars, v, I)
#     elseif v isa MPOTensor
#         delete!(t.scalars.data, I)
#         setindex!(t.tensors, v, I)
#     else
#         throw(ArgumentError("Cannot `setindex!` with type $(typeof(v))"))
#     end
#     return t
# end

# Base.firstindex(A::SparseMPOTensor, args...) = firstindex(A.tensors, args...)
# Base.lastindex(A::SparseMPOTensor, args...) = lastindex(A.tensors, args...)

# function Base.show(io::IO, t::SparseMPOTensor)
#     if get(io, :compact, false)
#         print(io, "SparseMPOTensor(", space(t), ")")
#         return nothing
#     end
#     println(io, "SparseMPOTensor(", space(t), "):")
#     println(io, "tensors: ", t.tensors)
#     print(io, "scalars: ", t.scalars)
#     return nothing
# end
# # function Base.getindex(t::SparseMPOTensor{S,T,E}, I::Vararg{Int,4}) where {S,T,E}
# #     if haskey(t.tensors, CartesianIndex(I...))
# #         return t.tensors[CartesianIndex(I...)]
# #     else
# #         τ = BraidingTensor{spacetype(T),storagetype(T)}(getsubspace(space(t), I...))
# #     end
# # end

# # Promotion
# # ---------
# function Base.promote_rule(
#     ::Type{<:SparseMPOTensor{S,<:AbstractTensorMap{S,N₁,N₂}}},
#     ::Type{<:AbstractTensorMap{S,N₁,N₂}},
# ) where {S,N₁,N₂}
#     return BlockTensorMap{S,N₁,N₂}
# end

# AbstractTensorMap Interface
# ---------------------------
# TensorKit.space(t::SparseMPOTensor) = space(t.tensors)
# TensorKit.space(t::SparseMPOTensor, i::Int) = space(t.tensors, i)

# TensorKit.storagetype(::Type{<:SparseMPOTensor{<:Any,T}}) where {T} = storagetype(T)

# TensorOperations Interface
# --------------------------
# function TensorOperations.tensorcontract!(
#     C::BlockTensorMap{S},
#     pC::Index2Tuple,
#     A::BlockTensorMap{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::SparseMPOTensor{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     α::Number,
#     β::Number,
# ) where {S}
#     tensorcontract!(C, pC, A, pA, conjA, B.tensors, pB, conjB, α, β)
#     for (IB, scalar) in nonzero_pairs(B.scalars)
#         τ = B[IB]
#         for IA in eachindex(A)
#             TupleTools.getindices(IA.I, pA[2]) == TupleTools.getindices(IB.I, pB[1]) ||
#                 continue
#             IC_unpermuted = (
#                 TupleTools.getindices(IA.I, pA[1])..., TupleTools.getindices(IB.I, pB[2])...
#             )
#             IC = CartesianIndex(TupleTools.getindices(IC_unpermuted, linearize(pC)))
#             tensorcontract!(C[IC], pC, A[IA], pA, conjA, τ, pB, conjB, scalar, One())
#         end
#     end
#     return C
# end
# function TensorOperations.tensorcontract!(
#     C::BlockTensorMap{S},
#     pC::Index2Tuple,
#     A::SparseMPOTensor{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::BlockTensorMap{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     α::Number,
#     β::Number,
# ) where {S}
#     tensorcontract!(C, pC, A.tensors, pA, conjA, B, pB, conjB, α, β)
#     for (IA, scalar) in nonzero_pairs(A.scalars)
#         τ = A[IA]
#         for IB in eachindex(B)
#             TupleTools.getindices(IA.I, pA[2]) == TupleTools.getindices(IB.I, pB[1]) ||
#                 continue
#             IC_unpermuted = (
#                 TupleTools.getindices(IA.I, pA[1])..., TupleTools.getindices(IB.I, pB[2])...
#             )
#             IC = CartesianIndex(TupleTools.getindices(IC_unpermuted, linearize(pC)))
#             tensorcontract!(C[IC], pC, τ, pA, conjA, B[IB], pB, conjB, scalar, One())
#         end
#     end
#     return C
# end

# function TensorOperations.tensorcontract_type(
#     TC::Type{<:Number},
#     pC::Index2Tuple{N₁,N₂},
#     A::SparseMPOTensor{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::AbstractTensorMap{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     istemp=false,
#     backend::TensorOperations.Backend...,
# ) where {S,N₁,N₂}
#     return TensorOperations.tensorcontract_type(TC, pC, A.tensors, pA, conjA, B, pB, conjB, istemp, backend...)
# end

# function TensorOperations.tensorcontract_type(
#     TC::Type{<:Number},
#     pC::Index2Tuple{N₁,N₂},
#     A::AbstractTensorMap{S},
#     pA::Index2Tuple,
#     conjA::Symbol,
#     B::SparseMPOTensor{S},
#     pB::Index2Tuple,
#     conjB::Symbol,
#     istemp=false,
#     backend::TensorOperations.Backend...,
# ) where {S,N₁,N₂}
#     return TensorOperations.tensorcontract_type(
#         TC, pC, A, pA, conjA, B.tensors, pB, conjB, istemp, backend...
#     )
# end

# Utility
# -------
Base.isone(x::SparseMPOTensor, I::Vararg{Int,4}) = isone(x.scalars[I...])
function Base.iszero(x::SparseMPOTensor, I::Vararg{Int,4})
    return iszero(x.scalars[I...]) && !haskey(x.tensors.data.data, CartesianIndex(I))
end
