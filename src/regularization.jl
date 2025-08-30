mutable struct Regularizers{T<:AbstractFloat}
    o_reg::Function
    wf_reg::Function
    λ_reg::Function
    βo::T
    βwf::T
    βλ::T
    βo_schedule::Function
    βwf_schedule::Function
    βλ_schedule::Function
    function Regularizers(;
            o_reg=no_reg(),
            wf_reg=no_reg(),
            λ_reg=no_reg(),
            βo=0.0,
            βwf=0.0,
            βλ=0.0,
            βo_schedule=ConstantSchedule(βo),
            βwf_schedule=ConstantSchedule(βwf),
            βλ_schedule=ConstantSchedule(βλ),
            verb=true,
            FTYPE=Float64
        )
        regularizers = new{FTYPE}(o_reg, wf_reg, λ_reg, βo, βwf, βλ, βo_schedule, βwf_schedule, βλ_schedule)
        if verb == true
            display(regularizers)
        end
        return regularizers
    end
end

function no_reg()
    ## Do nothing to g, return 0
    function no_reg(x, g, β; mask=[])
        return 0
    end

    return no_reg
end

# @views function tv2_reg(dim; FTYPE=Float64)
#     sx = zeros(FTYPE, dim, dim)
#     sy = zeros(FTYPE, dim, dim)
#     G = zeros(FTYPE, dim, 2*dim)
#     sx[1, 1] = -1
#     sx[1, end] = 1
#     sy[1, 1] = -1
#     sy[end, 1] = 1
#     mask = ones(FTYPE, dim, dim)
#     function fg(x, g, β; mask=mask)
#         G[:, 1:dim] .= mask .* conv_psf(x, sx)
#         G[:, dim+1:end] .= mask .* conv_psf(x, sy)
#         ϵ = FTYPE(β*norm(G, 2)^2)
#         g .+= FTYPE.(2*β .* (ccorr_psf(G[:, 1:dim], sx) .+ ccorr_psf(G[:, dim+1:end], sy)))
#         return ϵ
#     end

#     return fg
# end

@views function tv2_reg(dim; FTYPE=Float64)
    G = zeros(FTYPE, dim, dim, 2)
    mask = ones(FTYPE, dim, dim)
    function tv2_reg(x, g, β; mask=mask)
        G[:, 1:end-1, 1] .= x[:, 2:end] .- x[:, 1:end-1]
        G[:, end, 1] .= x[:, end-1] .- x[:, end]

        G[1:end-1, :, 2] .= x[2:end, :] .- x[1:end-1, :]
        G[end, :, 2] .= x[end-1, :] .- x[end, :]
        
        ϵ = FTYPE(β*norm(G, 2)^2)
        g[:, 1:end-1] .+= -2*β .* G[:, 1:end-1, 1]
        g[:, end] .+= 2*β .* G[:, end, 1]
        g[1:end-1, :] .+= -2*β .* G[1:end-1, :, 2]
        g[end, :] .+= 2*β .* G[end, :, 2] 
        return ϵ
    end

    return tv2_reg
end

@views function λtv_reg(dim, λ; FTYPE=Float64)
    nλ = length(λ)
    Δλ = FTYPE((maximum(λ) - minimum(λ)) / (nλ - 1))
    ∇O = zeros(FTYPE, dim, dim, nλ)
    function λtv_reg(x, g, β; mask=[])
        ∇O[:, :, 1:end-1] .= (x[:, :, 2:end] .- x[:, :, 1:end-1]) / Δλ
        ∇O[:, :, end] .= (x[:, :, end-1] .- x[:, :, end]) / Δλ
        ϵ = FTYPE(β*norm(∇O, 2)^2)
        g[:, :, 1:end-1] .+= -2 * β .* ∇O[:, :, 1:end-1]
        g[:, :, end] .+= 2 * β .* ∇O[:, :, end]
        return ϵ
    end

    return λtv_reg
end

@views function l2_reg(; FTYPE=Float64)
    function l2_reg(x, g, β; mask=[])
        ϵ = FTYPE(β*norm(x, 2)^2)
        g .+= FTYPE(2*β) .* x
        return ϵ
    end

    return l2_reg
end