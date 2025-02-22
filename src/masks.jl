using FITSIO
using Crayons

abstract type AbstractMasks end
function Base.display(masks::T) where {T<:AbstractMasks}
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Masks\n"); print(Crayon(reset=true))
    println("\tSize: $(masks.dim)×$(masks.dim) pixels")
    println("\tConfiguration: $(masks.nsubaps_side)×$(masks.nsubaps_side) subapertures")
    println("\tWavelength: $(minimum(masks.λ)) — $(maximum(masks.λ)) m")
    println("\tNumber of wavelengths: $(length(masks.λ)) wavelengths")
end

mutable struct Masks{T<:AbstractFloat} <: AbstractMasks
    masks::Array{T, 4}
    dim::Int64
    λ::Vector{T}
    λ_nyquist::T
    nλ::Int64
    Δλ::T
    nsubaps::Int64    
    nsubaps_side::Int64
    scale_psfs::Vector{T}
    ix::Vector{Int64}
    """
        Masks(dim, λ; nsubaps_side=..., λ_nyquist=..., verb=..., FTYPE=...)

    Create `Masks` struct. Mask arrays are created by [`HyperspeckleSpeckle.create_ish_masks`](@ref).

    * `verb` can be specified to print information on the masks to the terminal
    * `FTYPE` can be specified to change the Floating-point precision of the masks
    """
    function Masks(
            dim, 
            λ;
            nsubaps_side=1,
            λ_nyquist=minimum(λ),
            verb=true,
            FTYPE=Float64
        )
        masks_arr, ix = make_ish_masks(dim, nsubaps_side, λ, λ_nyquist=λ_nyquist, verb=false, FTYPE=FTYPE)
        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)
        nsubaps = size(masks_arr, 3)
        scale_psfs = [FTYPE(1 / norm(masks_arr[:, :, 1, w], 2)) for w=1:nλ]
        masks = new{FTYPE}(masks_arr, dim, λ, λ_nyquist, nλ, Δλ, nsubaps, nsubaps_side, scale_psfs, ix)
        if verb == true
            display(masks)
        end
        return masks
    end
end

"""
    mask = make_simple_mask(dim, D; FTYPE=...)

Create circular mask of diameter `D` in pixels within an array of size `dim` by `dim`.

* `FTYPE` can be specified to change the Floating-point precision of the mask
"""
@views function make_simple_mask(dim, D, FTYPE=Float64)
    nn = dim÷2 + 1
    x = collect(1:dim) .- nn
    rr = hypot.(x, x')
    mask = zeros(FTYPE, dim, dim)
    mask[rr .<= D/2] .= 1
    return mask
end

@views function make_ish_masks(dim, nsubaps_side, λ::T; λ_nyquist=minimum(λ), verb=true, FTYPE=Float64) where {T<:AbstractFloat}
    if verb == true
        print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Mask\n"); print(Crayon(reset=true))
        println("\tSize: $(dim)×$(dim) pixels")
        println("\tConfiguration: $(nsubaps_side)×$(nsubaps_side) subapertures")
        println("\tWavelength: $(λ) nm")
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
    mask_transform = ((transform+((dim÷2, dim÷2)))*(1/scaleby_wavelength)) - (dim÷2, dim÷2)
    scale_mask = TwoDimensionalTransformInterpolator(image_size, image_size, kernel, mask_transform)
    for n=1:nsubaps_side^2
        fill!(temp_mask, zero(FTYPE))
        xrange = subaperture_coords[n, 1][1]:subaperture_coords[n, 1][2]
        yrange = subaperture_coords[n, 2][1]:subaperture_coords[n, 2][2]        
        temp_mask[xrange, yrange] .= 1
        temp_mask .*= nyquist_mask
        subaperture_masks[:, :, n] = scale_mask*temp_mask
    end
    
    return subaperture_masks
end

"""
    subaperture_masks, ix = maks_ish_masks(dim, nsubaps_side, λ; λ_nyquist=..., verb=..., FTYPE=...)

Creates pupil masks of size `dim` by `dim` at wavelengths `λ`. If given `nsubaps_side`>1 it will create square
pupil masks for a Shack-Hartmann wavefront sensor. Masks will be nyquist sampled at `λ_nyquist`, meaning the 
pupil diameter will be exactly half the size of the array. Also returns the linear indices `ix` of an 
`nsubaps_side` by `nsubaps_side` grid that removes corner subaps with little representation. For 6 by 6
subaps, `ix` will start with indices 1:36 and then exclude 1, 6, 30, and 36.

* `verb` can be specified to print information on the masks to the terminal
* `FTYPE` can be specified to change the Floating-point precision of the mask
"""
@views function make_ish_masks(dim, nsubaps_side, λ::Vector{<:AbstractFloat}; λ_nyquist=minimum(λ), verb=true, FTYPE=Float64)
    if verb == true
        print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Masks\n"); print(Crayon(reset=true))
        println("\tSize: $(dim)×$(dim) pixels")
        println("\tConfiguration: $(nsubaps_side)×$(nsubaps_side) subapertures")
        println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
        println("\tNumber of wavelengths: $(length(λ)) wavelengths")
    end
    nλ = length(λ)
    subaperture_masks = Array{FTYPE, 4}(undef, dim, dim, nsubaps_side^2, nλ)
    Threads.@threads :static for w=1:nλ
        subaperture_masks[:, :, :, w] .= make_ish_masks(dim, nsubaps_side, λ[w], λ_nyquist=λ_nyquist, verb=false, FTYPE=FTYPE)
    end

    subap_flux = (((dim÷2) * λ_nyquist/λ[1]) / nsubaps_side)^2
    keepix = zeros(Bool, nsubaps_side^2)
    for n=1:nsubaps_side^2
        flux = sum(subaperture_masks[:, :, n, 1], dims=(1, 2))[1, 1]
        if (flux / subap_flux >= 0.5) || (nsubaps_side == 1)
            keepix[n] = 1
        end
    end

    ix = findall(keepix .== 1)
    return subaperture_masks[:, :, keepix, :], ix
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
