using ZernikePolynomials


mutable struct Diversity{T<:AbstractFloat}
    ixNoll::Int64
    waves::T
    period::T
    schedule::Function
    phase::Array{T, 3}
    function Diversity(ixNoll, waves, period, t0; verb=true, FTYPE=Float64)
        schedule = triangle_wave(t0, period, FTYPE=FTYPE)
        diversity = new{FTYPE}(ixNoll, waves, period, schedule)
        if verb == true
            display(diversity)
        end
        return diversity
    end
end

@views function create_diversity_phase!(diversity, masks)
    FTYPE = gettype(diversity)
    radius = masks.dim÷4 .* (masks.λ_nyquist ./ masks.λ)
    diversity.phase = zeros(FTYPE, masks.dim, masks.dim, masks.nλ)
    for w=1:masks.nλ
        diversity.phase[:, :, w] .= create_zernike_screen(masks.dim, radius[w], diversity.ixNoll, diversity.waves, FTYPE=FTYPE)
    end
end

function create_zernike_screen(dim, radius, index, waves; FTYPE=Float64)
    x = collect(-dim÷2:dim÷2-1) ./ radius
    ϕz = evaluatezernike(x, x, [Noll(index)], [waves])
    N = normalization(Noll(index))
    ϕz .*= pi / N
    return FTYPE.(ϕz)
end

function triangle_wave(t0, P; FTYPE=Float64)
    return (t) -> FTYPE(2 * abs(2*((t - t0) / P - floor((t - t0) / P + 1/2))) - 1)
end
