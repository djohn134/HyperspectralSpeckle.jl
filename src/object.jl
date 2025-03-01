using FITSIO
using Statistics
using NumericalIntegration
using Interpolations: interpolate, Gridded, Linear


abstract type AbstractObject end
function Base.display(object::T) where {T<:AbstractObject}
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Object\n"); print(Crayon(reset=true))
    println("\tSize: $(object.dim)×$(object.dim) pixels")
    println("\tFOV: $(object.fov)×$(object.fov) arcsec")
    println("\tRange: $(object.range) m")
    println("\tIrradiance: $(object.irradiance) ph/s/m^2")
    println("\tBackground Irradiance: $(object.background) ph/s/m^2")
    println("\tWavelength: $(minimum(object.λ)) — $(maximum(object.λ)) m")
    println("\tNumber of wavelengths: $(length(object.λ))")
end

mutable struct Object{T<:AbstractFloat} <: AbstractObject
    dim::Int64
    λ::Vector{T}
    nλ::Int64
    Δλ::T
    range::T
    fov::T
    sampling_arcsecperpix::T
    spectrum::Vector{T}
    irradiance::T
    background::T
    object::Array{T, 3}
    function Object(
            object_arr;
            irradiance=Inf,
            background=0,
            λ=[Inf], 
            dim=0, 
            fov=0,
            object_range=0,
            spectrum=[0],
            scaled=false,
            FTYPE=Float64,
            verb=true
        )
        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)
        sampling_arcsecperpix = fov / dim
        if scaled == false
            for w=1:nλ
                object_arr[:, :, w] .*= spectrum[w]
            end
            object_arr ./= sum(object_arr)
            object_arr .*= irradiance / Δλ
        end
        object = new{FTYPE}(dim, λ, nλ, Δλ, object_range, fov, sampling_arcsecperpix, spectrum, irradiance, background, object_arr)
        if verb == true
            display(object)
        end
        return object
    end
end

function mag2flux(mag, filter; ζ=0.0)
    ## Flux at top of atmosphere
    airmass = secd(ζ)
    irradiance_vega = magnitude_zeropoint(filter.λ, filter.response)  # ph/s/m^2
    # radiant_energy_target = exptime * area * irradiance_vega * 10^(-(mag + 0.3*airmass) / 2.5)  # ph
    irradiance_target = irradiance_vega * 10^(-(mag + 0.3*airmass) / 2.5)  # ph/s/m^2
    # return radiant_energy_target
    return irradiance_target
end

function magnitude_zeropoint(λfilter, filter_response)
    # Photons per square meter per second produced by a 0th mag star above the atmosphere.
    # Assuming spectrum like Vega
    h = 6.63e-34  # J s
    c = 3e8  # m / s

    ~, spectral_irradiance = vega_spectrum(λ=λfilter)  # W/m^2/m
    spectral_irradiance ./= h*c ./ λfilter  # ph/s/m^2/m
    irradiance = NumericalIntegration.integrate(λfilter, spectral_irradiance .* filter_response)  # ph/s/m^2
    return irradiance
end

@views function template2object(template, dim, λ; FTYPE=Float64)
    nλ = length(λ)
    object = zeros(Float64, dim, dim, nλ)
    nmaterials = 6
    materials = Array{Array{Float64}}(undef, nmaterials)
    # Solar Panel (AR coated)
    materials[1] = [9.667504e2, -4.583580e0, 8.008073e-3, -6.110959e-6, 1.73911e-9]
    # Kapton
    materials[2] = [-1.196472E5, 1.460390E3, -7.648047E0, 2.246897E-2, -4.056309E-5, 4.615807E-8, -3.238676E-11, 1.283035E-14, -2.200156E-18]
    # Aluminized Mylar
    materials[3] = [-7.964498E3, 7.512566E1, -2.883463E-1, 5.812354E-4, -6.488131E-7, 3.801200E-10, -9.129042E-14]
    # Kapton - aged:4
    materials[4] = [-7.668973E4, 9.501756E2, -5.055507E0, 1.509819E-2, -2.771163E-5, 3.204811E-8, -2.283223E-11, 9.171536E-15, -1.591887E-18]
    # Aluminized Mylar - aged:4
    materials[5] = [-2.223456E4, 2.305905E2, -1.007697E0, 2.404597E-3, -3.379637E-6, 2.797231E-9, -1.262900E-12, 2.401227E-16]
    # Faint solar panels 
    materials[6] = 1e-2*materials[1];
    # Read in data image (FITS format)
    object_coeffs = Int.(crop(readfits(template), dim))
    spectra = hcat([[sum([materials[i][ll+1] * (λ[k].*1e9).^ll for ll in 0:length(materials[i])-1]) for k = 1:nλ] for i=1:nmaterials]...) ./ 100

    for i = 1:nmaterials    
        indx = findall(object_coeffs .== i)
        for j in indx
            object[j, :] .= spectra[:, i]
        end
    end

    return FTYPE.(object), FTYPE.(spectra)
end

function interpolate_object(object, λin, λout)
    x = 1:size(object, 1)
    y = 1:size(object, 2)
    itp = interpolate((x, y, λin), object, Gridded(Linear()))
    return itp(x, y, λout)
end

function poly2object(coeffs::AbstractArray{<:AbstractFloat, 3}, λ; FTYPE=Float64)
    dim = size(coeffs, 1)
    nλ = length(λ)
    object = zeros(FTYPE, dim, dim, nλ)
    poly2object!(object, coeffs, λ)
    return object
end

@views function poly2object!(object, coeffs::AbstractArray{<:AbstractFloat, 3}, λ)
    ncoeffs = size(coeffs, 3)
    nλ = length(λ)
    for w=1:nλ
        for k=1:ncoeffs
            object[:, :, w] .+= λ[w]^(k-1) .* coeffs[:, :, k]
        end
    end
end

function object2poly(object, λ, ncoeffs; FTYPE=Float64)
    dim = size(object, 1)
    coeffs = zeros(FTYPE, dim, dim, ncoeffs)
    object2poly!(coeffs, ncoeffs, object, λ)
    return coeffs
end

@views function object2poly!(coeffs, ncoeffs, object, λ)
    nonzero = findall(dropdims(sum(object, dims=3), dims=3) .> 0)
    for np in nonzero
        coeffs[np, :] .= poly_fit(λ, object[np, :], ncoeffs-1)
    end
end

@views function fit_background(observations)
    FTYPE = gettype(observations)
    backgrounds = zeros(FTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs)
    mask = ones(FTYPE, observations.dim, observations.dim)
    for n=1:observations.nsubaps
        for t=1:observations.nepochs
            for ~=1:10
                backgrounds[:, :, n, t] .= fit_plane(observations.images[:, :, n, t], mask)
                res = observations.images[:, :, n, t] .- backgrounds[:, :, n, t]
                σ = std(res)
                mask[res .> σ] .= FTYPE(0.0)
            end
            fill!(mask, FTYPE(1.0))
        end
    end

    return Statistics.mean(dropdims(sum(backgrounds, dims=(1, 2)), dims=(1, 2)))
end
