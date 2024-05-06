using FITSIO


mutable struct Masks{T<:AbstractFloat}
    masks::Array{T, 4}
    dim::Int64
    λ::Vector{T}
    λ_nyquist::T
    nλ::Int64
    Δλ::T
    nsubaps::Int64
    scale_psfs::Vector{T}
    function Masks(;
            maskfile="",
            dim=256,
            nsubaps_side=1,
            λ=[400.0],
            λ_nyquist=400.0,
            FTYPE=Float64
        )
        if (maskfile != "")
            masks, λ = readmasks(maskfile, FTYPE=FTYPE)
            dim = size(masks, 1)
        else
            masks = make_ish_masks(dim, nsubaps_side, λ, λ_nyquist=λ_nyquist, FTYPE=FTYPE)
        end

        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)

        nsubaps = size(masks, 3)
        scale_psfs = [FTYPE(1 / norm(masks[:, :, 1, w], 2)) for w=1:nλ]
        new{FTYPE}(masks, dim, λ, λ_nyquist, nλ, Δλ, nsubaps, scale_psfs)
    end
end

@views function make_simple_mask(dim, D, FTYPE=Float64)
    nn = dim÷2 + 1
    x = collect(1:dim) .- nn
    rr = hypot.(x, x')
    mask = zeros(FTYPE, dim, dim)
    mask[rr .<= D/2] .= 1
    return mask
end

@views function make_ish_masks(dim, nsubaps_side, λ::T; λ_nyquist=400.0, verb=true, FTYPE=Float64) where {T<:AbstractFloat}
    if verb == true
        println("Creating $(dim)×$(dim) mask for $(nsubaps_side)×$(nsubaps_side) subapertures at $(λ) nm")
    end
    rad_nyquist = dim ÷ 4
    scaleby_wavelength = λ_nyquist/λ
    
    nn = dim÷2 + 1
    x = collect(1:dim) .- nn
    rr = hypot.(x, x')
    nyquist_mask = zeros(FTYPE, dim, dim)
    nyquist_mask[rr .<= rad_nyquist] .= 1
    temp_mask = zeros(FTYPE, dim, dim)

    npix_subap = round(Int64, 2*rad_nyquist / nsubaps_side)
    subaperture_masks = zeros(FTYPE, dim, dim, nsubaps_side^2)
    xstarts = round.(Int64, (nn-rad_nyquist).+[0:nsubaps_side-1;]*npix_subap.+1)
    xends = xstarts .+ (npix_subap-1)
    xpos = [[xstarts[n], xends[n]] for n=1:nsubaps_side];
    subaperture_coords = hcat([i for i in xpos for j in xpos], [j for i in xpos for j in xpos]);

    kernel = LinearSpline(FTYPE)
    transform = AffineTransform2D{FTYPE}()

    image_size = (Int64(dim), Int64(dim))
    for n=1:nsubaps_side^2
        fill!(temp_mask, zero(FTYPE))
        xrange = subaperture_coords[n, 1][1]:subaperture_coords[n, 1][2]
        yrange = subaperture_coords[n, 2][1]:subaperture_coords[n, 2][2]        
        temp_mask[xrange, yrange] .= 1
        temp_mask .*= nyquist_mask

        mask_transform = ((transform+((dim÷2, dim÷2)))*(1/scaleby_wavelength)) - (dim÷2, dim÷2)
        scale_mask = TwoDimensionalTransformInterpolator(image_size, image_size, kernel, mask_transform)
        subaperture_masks[:, :, n] = scale_mask*temp_mask
    end
    
    return subaperture_masks
end

@views function make_ish_masks(dim, nsubaps_side, λ::Vector{<:AbstractFloat}; λ_nyquist=400.0, FTYPE=Float64)
    nλ = length(λ)
    subaperture_masks = Array{FTYPE, 4}(undef, dim, dim, nsubaps_side^2, nλ)
    Threads.@threads :static for w=1:nλ
        subaperture_masks[:, :, :, w] .= make_ish_masks(dim, nsubaps_side, λ[w], λ_nyquist=λ_nyquist, FTYPE=FTYPE)
    end

    subap_flux = (((dim÷2) * λ_nyquist/λ[1]) / nsubaps_side)^2
    keepix = zeros(Bool, nsubaps_side^2)
    for n=1:nsubaps_side^2
        flux = sum(subaperture_masks[:, :, n, 1], dims=(1, 2))[1, 1]
        if (flux / subap_flux >= 0.5) || (nsubaps_side == 1)
            keepix[n] = 1
        end
    end

    return subaperture_masks[:, :, keepix, :]
end

@views function readmasks(files; FTYPE=Float64)
    nfiles = length(files)
    masks = Array{Array{FTYPE}}(undef, nfiles)
    for i=1:nfiles
        masks[i] = readfits(files[i], FTYPE=FTYPE)
    end

    hdu = FITS(files[1])[1]
    λstart = read_key(hdu, "WAVELENGTH_START")[1]
    λend = read_key(hdu, "WAVELENGTH_END")[1]
    Nλ = read_key(hdu, "WAVELENGTH_STEPS")[1]
    λ = FTYPE.(collect(range(λstart, stop=λend, length=Nλ)))

    masks = cat(masks..., dims=3)
    nframes = size(masks, 3)
    weights::Vector{FTYPE} = [FTYPE(sum(masks[:, :, n, 1])/sum(masks[:, :, end, 1])) for n=1:nframes]
    return masks, λ, weights
end
