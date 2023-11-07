"
    SparseMPO - used to represent both time evolution mpos and hamiltonians
"

struct InfiniteMPO{T<:AbstractMPOTensor} <: AbstractMPO
    data::PeriodicArray{T,1}
    function InfiniteMPO{T}(data::PeriodicArray{T,1}) where {T<:AbstractMPOTensor}
        for i in eachindex(data)
            left_virtualspace(data[i]) == dual(right_virtualspace(data[i - 1])) ||
                throw(TensorKit.SpaceMismatch("Incompatible spaces at $i"))
        end
        return new{T}(data)
    end
end

const SparseMPO{S,T} = InfiniteMPO{T} where {S,T<:SparseMPOTensor{S}}

VectorInterface.scalartype(::Type{<:InfiniteMPO{T}}) where {T} = scalartype(T)

left_virtualdim(O::SparseMPO, i::Int) = size(O[i], 1)
right_virtualdim(O::SparseMPO, i::Int) = size(O[i], 4)

# struct SparseMPO{S,T<:MPOTensor,E<:Number} <: AbstractVector{SparseMPOTensor{S,T,E}}
#     data::PeriodicArray{SparseMPOTensor{S,T,E},1}
# end

# struct SparseMPO{S,T<:MPOTensor,E<:Number} <: AbstractVector{SparseMPOSlice{S,T,E}}
#     Os::PeriodicArray{Union{E,T},3}
#     domspaces::PeriodicArray{S,2}
#     pspaces::PeriodicArray{S,1}
# end

function Base.getproperty(h::SparseMPO, f::Symbol)
    if f == :odim
        return size(h.domspaces, 2)
    elseif f == :period
        return size(h.pspaces, 1)
    elseif f == :imspaces
        return circshift(adjoint.(h.domspaces), (-1, 0))
    else
        return getfield(h, f)
    end
end

Base.eltype(::InfiniteMPO{T}) where {T} = T
Base.eltype(::Type{InfiniteMPO{T}}) where {T} = T
Base.parent(O::InfiniteMPO) = O.data
period(O::InfiniteMPO) = length(parent(O))
TensorKit.spacetype(O::Union{InfiniteMPO, Type{<:InfiniteMPO}}) = spacetype(eltype(O))

Base.checkbounds(a::SparseMPO, I...) = true

# AbstractArray
# -------------
Base.parent(a::SparseMPO) = a.data
Base.size(a::SparseMPO) = size(parent(a))
Base.length(a::SparseMPO) = length(parent(a))
Base.getindex(a::SparseMPO, i::Int) = getindex(parent(a), i)
Base.setindex!(a::SparseMPO, v, i::Int) = setindex!(parent(a), v, i)
Base.copy(x::SparseMPO) = SparseMPO(copy(parent(x)))
Base.deepcopy(x::SparseMPO) = SparseMPO(deepcopy(parent(x)))

# promotion and conversion
# ------------------------
function Base.promote_rule(::Type{InfiniteMPO{T₁}}, ::Type{InfiniteMPO{T₂}}) where {T₁,T₂}
    return InfiniteMPO{promote_type(T₁, T₂)}
end

function Base.convert(::Type{InfiniteMPO{T₁}}, x::InfiniteMPO{T₂}) where {T₁,T₂}
    T₁ === T₂ && return x
    return InfiniteMPO{T₁}(map(Base.Fix1(convert, T₁), x.data))
end


#=
allow passing in
        - non strictly typed matrices
        - missing fields
        - 2leg tensors
        - only mpo tensors
=#

# bit of a helper - accept non strict typed data
SparseMPO(x::AbstractArray{Any,3}) = SparseMPO(union_split(x));

#another helper - artificially create a union and reuse next constructor
function SparseMPO(x::AbstractArray{T,3}) where {T<:TensorMap}
    return SparseMPO(convert(AbstractArray{Union{T,scalartype(T)},3}, x))
end

function SparseMPO(x::AbstractArray{T,3}) where {T<:Union{A}} where {A}
    (Sp, M, E) = _envsetypes(union_types(T))

    nx = similar(x, Union{E,M})

    for (i, t) in enumerate(x)
        if t isa MPSBondTensor
            nx[i] = add_util_leg(t)
        elseif ismissing(t)
            nx[i] = zero(E)
        elseif t isa Number
            nx[i] = convert(E, t)
        else
            nx[i] = t
        end
    end

    return SparseMPO(nx)
end

#default constructor
function SparseMPO(x::AbstractArray{Union{E,M},3}) where {M<:MPOTensor,E<:Number}
    (period, numrows, numcols) = size(x)

    Sp = spacetype(M)
    E == scalartype(M) || throw(
        ArgumentError("scalar type should match mpo scalartype $E ≠ $(scalartype(M))")
    )
    numrows == numcols || throw(ArgumentError("mpos have to be square"))

    domspaces = PeriodicArray{Union{Missing,Sp}}(missing, period, numrows)
    pspaces = PeriodicArray{Union{Missing,Sp}}(missing, period)

    isused = fill(false, period, numrows, numcols)
    isstopped = false
    while !isstopped
        isstopped = true

        for i in 1:period, j in 1:numrows, k in 1:numcols
            isused[i, j, k] && continue

            if x[i, j, k] isa MPOTensor
                isused[i, j, k] = true
                isstopped = false

                #asign spaces when possible
                dom = _firstspace(x[i, j, k])
                im = _lastspace(x[i, j, k])
                p = space(x[i, j, k], 2)

                ismissing(pspaces[i]) && (pspaces[i] = p)
                pspaces[i] != p && throw(
                    ArgumentError(
                        "physical space for $((i,j,k)) incompatible : $(pspaces[i]) ≠ $(p)",
                    ),
                )

                ismissing(domspaces[i, j]) && (domspaces[i, j] = dom)
                domspaces[i, j] != dom && throw(
                    ArgumentError(
                        "Domspace for $((i,j,k)) incompatible : $(domspaces[i,j]) ≠ $(dom)",
                    ),
                )

                ismissing(domspaces[i + 1, k]) && (domspaces[i + 1, k] = im')
                domspaces[i + 1, k] != im' && throw(
                    ArgumentError(
                        "Imspace for $((i,j,k)) incompatible : $(domspaces[i+1,k]) ≠ $(im')",
                    ),
                )

                #if it's zero -> store zero
                #if it's the identity -> store identity
                if x[i, j, k] ≈ zero(x[i, j, k])
                    x[i, j, k] = zero(E) #the element is zero/missing
                else
                    ii, sc = isid(x[i, j, k])

                    if ii #the tensor is actually proportional to the identity operator -> store this knowledge
                        x[i, j, k] = sc ≈ one(sc) ? one(sc) : sc
                    end
                end
            elseif x[i, j, k] != zero(E)
                if !ismissing(domspaces[i, j])
                    isused[i, j, k] = true
                    isstopped = false

                    ismissing(domspaces[i + 1, k]) &&
                        (domspaces[i + 1, k] = domspaces[i, j])
                    domspaces[i + 1, k] != domspaces[i, j] && throw(
                        ArgumentError(
                            "Identity incompatible at $((i,j,k)) : $(domspaces[i+1,k]) ≠ $(domspaces[i,j])",
                        ),
                    )
                    _can_unambiguously_braid(domspaces[i, j]) ||
                        throw(ArgumentError("ambiguous identity operator $((i,j,k))"))
                elseif !ismissing(domspaces[i + 1, k])
                    isused[i, j, k] = true
                    isstopped = false

                    ismissing(domspaces[i, j]) && (domspaces[i, j] = domspaces[i + 1, k])
                    domspaces[i + 1, k] != domspaces[i, j] && throw(
                        ArgumentError(
                            "Identity incompatible at $((i,j,k)) : $(domspaces[i+1,k]) ≠ $(domspaces[i,j])",
                        ),
                    )
                    _can_unambiguously_braid(domspaces[i, j]) ||
                        throw(ArgumentError("ambiguous identity operator $((i,j,k))"))
                end

            else
                isused[i, j, k] = true
            end
        end
    end

    sum(ismissing.(pspaces)) == 0 ||
        throw(ArgumentError("Not all physical spaces were assigned"))
    
    
    replace!(domspaces, missing => oneunit(Sp))
    
    # for loc in 1:period, j in 1:numrows
    #     ismissing(domspaces[loc, j]) || continue
    #     domspaces[loc, j] = oneunit(Sp) # all(iszero.(x[loc,j,:])) ? zero(Sp) : oneunit(Sp)
    # end

    SumSpace.(identity.(domspaces[1, :]))
    ndomspaces = PeriodicArray{Sp}(domspaces)
    npspaces = PeriodicArray{Sp}(pspaces)

    blockt = map(1:period) do i
        P = SumSpace(npspaces[i])
        Vₗ = SumSpace(ndomspaces[i, :]...)
        Vᵣ = SumSpace(ndomspaces[mod1(i + 1, period), :]...)
        tdst = SparseMPOTensor(BlockTensorMap(undef_blocks, SparseArray{M,4}, Vₗ ⊗ P, P ⊗ Vᵣ))
        for j in axes(x, 2), k in axes(x, 3)
            tdst[j, 1, 1, k] = x[i, j, k]
        end
        return tdst
    end
    
    return SparseMPO{Sp,M,E}(PeriodicArray(blockt))
end

function _envsetypes(d::Tuple)
    a = Base.first(d)
    b = Base.tail(d)

    if a <: MPOTensor
        return spacetype(a), a, scalartype(a)
    elseif a <: MPSBondTensor
        return spacetype(a), tensormaptype(spacetype(a), 2, 2, scalartype(a)), scalartype(a)
    else
        @assert !isempty(b)
        return _envsetypes(b)
    end
end

# Base.size(x::SparseMPO) = (size(x.Os, 1),);
# function Base.getindex(x::SparseMPO{S,T,E}, a::Int) where {S,T,E}
#     return SparseMPOSlice{S,T,E}(
#         @view(x.Os[a, :, :]),
#         @view(x.domspaces[a, :]),
#         @view(x.imspaces[a, :]),
#         x.pspaces[a],
#     )
# end;

TensorKit.space(x::SparseMPO, i) = x.pspaces[i]
"
checks if ham[:,i,i] = 1 for every i
"
function isid(ham::SparseMPO, i::Int)
    E = scalartype(ham)
    return reduce(
        (a, b) -> a && isscal(ham, b, i, i) && abs(ham.Os[b, i, i] - one(E)) < 1e-14,
        1:(ham.period);
        init=true,
    )
end

"
checks if the given 4leg tensor is the identity (needed for infinite mpo hamiltonians)
"
function isid(x::MPOTensor; tol=Defaults.tolgauge)
    (_firstspace(x) == _lastspace(x)' && space(x, 2) == space(x, 3)') ||
        return false, zero(scalartype(x))
    _can_unambiguously_braid(_firstspace(x)) || return false, zero(scalartype(x))
    iszero(norm(x)) && return false, zero(scalartype(x))

    id = isomorphism(storagetype(x), space(x, 2), space(x, 2))
    @plansor t[-1; -2] := τ[3 -1; 1 2] * x[1 2; 3 -2]
    scal = tr(t) / dim(codomain(x))
    @plansor diff[-1 -2; -3 -4] := τ[-1 -2; 1 2] * (scal * one(t))[2; -4] * id[1; -3]
    diff -= x

    return norm(diff) < tol, scal
end

function Base.:*(b::SparseMPO{T}, a::SparseMPO{T}) where {T}
    error("not implemented")
    # TODO: implement
    # nodim = a.odim * b.odim
    # indmap = LinearIndices((a.odim, b.odim))
    # nOs = PeriodicArray{Union{E,T},3}(fill(zero(E), a.period, nodim, nodim))

    # fusers = PeriodicArray(
    #     map(product(1:(a.period), 1:(a.odim), 1:(b.odim))) do (pos, i, j)
    #         return isomorphism(
    #             storagetype(T),
    #             fuse(a.domspaces[pos, i] * b.domspaces[pos, j]),
    #             a.domspaces[pos, i] * b.domspaces[pos, j],
    #         )
    #     end,
    # )

    # ndomspaces = PeriodicArray{S,2}(undef, a.period, nodim)
    # for pos in 1:(a.period), i in 1:(a.odim), j in 1:(b.odim)
    #     ndomspaces[pos, indmap[i, j]] = codomain(fusers[pos, i, j])
    # end

    # for pos in 1:(a.period), (i, j) in keys(a[pos]), (k, l) in keys(b[pos])
    #     if isscal(a[pos], i, j) && isscal(b[pos], k, l)
    #         nOs[pos, indmap[i, k], indmap[j, l]] = a.Os[pos, i, j] * b.Os[pos, k, l]
    #     else
    #         @plansor nOs[pos, indmap[i, k], indmap[j, l]][-1 -2; -3 -4] :=
    #             fusers[pos, i, k][
    #                 -1
    #                 1 2
    #             ] *
    #             conj(fusers[pos + 1, j, l][
    #                 -4
    #                 3 4
    #             ]) *
    #             a[pos][i, j][
    #                 1 5
    #                 -3 3
    #             ] *
    #             b[pos][k, l][
    #                 2 -2
    #                 5 4
    #             ]
    #     end
    # end

    # return SparseMPO{S,T,E}(nOs, ndomspaces, a.pspaces)
end

# without the copy, we get side effects when repeating + setindex
function Base.repeat(x::InfiniteMPO, n::Int)
    return InfiniteMPO(repeat(x.data, n))
end

function Base.conj(a::InfiniteMPO)
    data′ = map(a.data) do x
        @plansor x′[-1 -2; -3 -4] := conj(x[-1 -3; -2 -4])
    end
    return InfiniteMPO(data′)
end

function Base.convert(::Type{DenseMPO}, s::SparseMPO)
    return InfiniteMPO(map(Base.Fix1(convert, MPOTensor), s.data))
    
    # embeds = PeriodicArray(_embedders.([s[i].domspaces for i in 1:length(s)]))
    
    # data = PeriodicArray(
    #     map(1:size(s, 1)) do loc
    #         return mapreduce(+, Iterators.product(1:(s.odim), 1:(s.odim))) do (i, j)
    #             return @plansor temp[-1 -2; -3 -4] :=
    #                 embeds[loc][i][-1; 1] *
    #                 s[loc][i, j][1 -2; -3 2] *
    #                 conj(embeds[loc + 1][j][-4; 2])
    #         end
    #     end,
    # )

    # #there are often 0-blocks, which we can just filter out
    # for i in 1:length(data)
    #     (U, S, V) = tsvd(
    #         transpose(data[i], ((3, 1, 2), (4,))); trunc=truncbelow(Defaults.tolgauge)
    #     )
    #     data[i] = transpose(U, ((2, 3), (1, 4)))
    #     @plansor data[i + 1][-1 -2; -3 -4] := S[-1; 1] * V[1; 2] * data[i + 1][2 -2; -3 -4]

    #     (U, S, V) = tsvd(
    #         transpose(data[i], ((1,), (3, 4, 2))); trunc=truncbelow(Defaults.tolgauge)
    #     )
    #     data[i] = transpose(V, ((1, 4), (2, 3)))
    #     @plansor data[i - 1][-1 -2; -3 -4] := data[i - 1][-1 -2; -3 1] * U[1; 2] * S[2; -4]
    # end

    # return DenseMPO(data)
end

"""
    remove_orphans(mpo::SparseMPO)

Prune all branches of the finite state machine that do not contribute. Additionally, attempt
to compact the representation as much as possible.
"""
function remove_orphans!(smpo::SparseMPO; tol=eps(real(scalartype(smpo)))^(3/4))
    # drop zeros
    for slice in parent(smpo)
        for (key, val) in BlockTensorKit.nonzero_pairs(slice)
            norm(val) < tol && delete!(slice, key)
        end
    end
    
    # drop dead starts/ends
    changed = true
    while changed
        changed = false
        
        for i in 1:length(smpo)
            # slice empty columns on right or empty rows on left
            mask = filter(1:size(smpo[i], 4)) do j
                return j ∈ getindex.(BlockTensorKit.nonzero_keys(smpo[i]), 1) ||
                    j ∈ getindex.(BlockTensorKit.nonzero_keys(smpo[i + 1]), 4)
            end
            changed |= length(mask) == size(smpo[i], 4)
            smpo[i] = smpo[i][:, :, :, mask]
            smpo[i + 1] = smpo[i + 1][mask, :, :, :]
        end
        
        # for (i, slice) in enumerate(parent(smpo))
        #     nz_keys = BlockTensorKit.nonzero_keys(slice)
            
        #     # remove rows of previous site that lead to dead end
        #     setdiff!(zero_cols[i], [x[4] for x in nz_keys])
        #     if !isempty(zero_cols[i])
        #         for key in BlockTensorKit.nonzero_keys(smpo[i - 1])
        #             if key[1] ∈ empty_col
        #                 delete!(smpo[i - 1], key)
        #                 changed = true
        #             end
        #         end
        #     end
            
        #     # remove cols of next site that come from dead end
        #     empty_rows = setdiff(1:size(slice, 1), [x[1] for x in nz_keys])
        #     if !isempty(empty_rows)
        #         for key in BlockTensorKit.nonzero_keys(smpo[i + 1])
        #             if key[4] ∈ empty_row
        #                 delete!(smpo[i + 1], key)
        #                 changed = true
        #             end
        #         end
        #     end
        # end
    end
    
    return smpo
    
    # # slice out empty rows and columns
    # for i in 1:length(smpo)
    #     empty_cols = setdiff(1:size(smpo[i], 4), [x[4] for x in nz_keys]
    # end
    
    # @warn "not implemented"
    # return smpo
    # changed = false # if I change the mpo somewhere in the method, then I will return remove_orphans(changed_mpo)

    # out = copy(smpo)
    # dead_ends = fill(true, out.odim)
    # dead_starts = fill(true, out.odim)

    # for (loc, slice) in enumerate(out)
    #     for i in 1:(out.odim)
    #         if all(slice.Os[i, :] .== zero(E)) # dead start
    #             changed |= !all(out[loc - 1].Os[:, i] .== zero(E))
    #             out[loc - 1].Os[:, i] .= zero(E)
    #         else
    #             dead_starts[i] = false
    #         end

    #         if all(slice.Os[:, i] .== zero(E)) # dead end
    #             changed |= !all(out[loc + 1].Os[i, :] .== zero(E))

    #             out[loc + 1].Os[i, :] .= zero(E)
    #         else
    #             dead_ends[i] = false
    #         end
    #     end
    # end

    # removeable = dead_ends .| dead_starts
    # if any(removeable)
    #     changed = true

    #     keep = .!removeable

    #     new_Os = PeriodicArray(out.Os[:, keep, keep])
    #     new_domspaces = PeriodicArray(out.domspaces[:, keep])
    #     new_pspaces = PeriodicArray(out.pspaces)

    #     out = SparseMPO(new_Os, new_domspaces, new_pspaces)
    # end

    # return changed ? remove_orphans(out) : out
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
                O_new[i][j, k][-1 -2; -3 -4] :=
                    F[-2; 1 2] * O[i][j, k][-1 1; 4 3] * τ[3 2; 5 -4] * conj(F[-3; 4 5])
            end
        end
    end

    return O_new
end
