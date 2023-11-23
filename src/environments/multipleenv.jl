struct MultipleEnvironments{O,C} <: Cache
    envs::Vector{C}
end

Base.size(x::MultipleEnvironments) = size(x.envs)
Base.getindex(x::MultipleEnvironments, i) = x.envs[i]
Base.length(x::MultipleEnvironments) = prod(size(x))

Base.iterate(x::MultipleEnvironments) = iterate(x.envs)
Base.iterate(x::MultipleEnvironments, i) = iterate(x.envs, i)

# we need constructor, agnostic of particular MPS
function environments(st, ham::LazySum)
    return MultipleEnvironments(ham, map(op -> environments(st, op), ham.ops))
end

#broadcast vs map?
# function environments(state, ham::LinearCombination)
#     return MultipleEnvironments(ham, broadcast(o -> environments(state, o), ham.opps))
# end;

function environments(
    st::WindowMPS,
    ham::LazySum;
    lenvs=environments(st.left_gs, ham),
    renvs=environments(st.right_gs, ham),
)
    return MultipleEnvironments(
        ham,
        map(
            (op, sublenv, subrenv) -> environments(st, op; lenvs=sublenv, renvs=subrenv),
            ham.ops,
            lenvs,
            renvs,
        ),
    )
end

# we need to define how to recalculate
"""
    Recalculate in-place each sub-env in MultipleEnvironments
"""
function recalculate!(env::MultipleEnvironments, args...)
    for subenv in env.envs
        recalculate!(subenv, args...)
    end
    return env
end
