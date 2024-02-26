"
    Window(left,middle,right)

    general struct of an object with a left, middle and right part.
"
struct Window{L,M,R}
    left::L
    middle::M
    right::R
end

# Holy traits
TimeDependence(x::Window) = istimed(x) ? TimeDependent() : NotTimeDependent()
istimed(x::Window) = istimed(x.left) || istimed(x.middle) || istimed(x.right)

_eval_at(x::Window, args...) =  Window(_eval_at(x.left,args...),_eval_at(x.middle,args...),_eval_at(x.right,args...))
safe_eval(::TimeDependent, x::Window, t::Number) = _eval_at(x,t)

# For users
(x::Window)(t::Number) = safe_eval(x, t)