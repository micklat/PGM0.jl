import NumericFuns: logsumexp

function logsumexp(x::AbstractArray)
    m = maximum(x);
    m .+ log(sum(exp(x .- m)))
end

function logsumexp(x::AbstractArray, axis::Int; squeeze=true)
    m = maximum(x, axis)
    res = log(sum(exp(x .- m), axis)) .+ m
    squeeze ? Base.squeeze(res, axis) : res
end
