using FITSIO
using Statistics
using NumericalIntegration
using Interpolations: interpolate, Gridded, Linear


mutable struct Object{T<:AbstractFloat}
    λ::Vector{T}
    nλ::Int64
    height::T
    fov::T
    sampling_arcsecperpix::T
    spectrum::Vector{T}
    flux::T
    background_flux::T
    object::Array{T, 3}
    function Object(; 
            flux=Inf,
            background_flux=0,
            λ=[Inf], 
            dim=0, 
            fov=0,
            height=0,
            spectrum=[0],
            objectfile="", 
            template=false, 
            FTYPE=Float64,
            verb=true
        )
        nλ = length(λ)
        Δλ = (nλ == 1) ? 1.0 : (maximum(λ) - minimum(λ)) / (nλ - 1)
        sampling_arcsecperpix = fov / dim
        if objectfile != ""
            if verb == true
                print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Object\n"); print(Crayon(reset=true))
                println("\tSize: $(dim)x$(dim) pixels")
                println("\tFOV: $(fov)×$(fov) arcsec")
                println("\tHeight: $(height) km")
                println("\tFlux: $(flux) ph")
                println("\tBackground flux: $(background_flux) ph")
                println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
                println("\tNumber of wavelengths: $(length(λ))")
            end
            if template == true
                object, ~ = template2object(objectfile, dim, λ, FTYPE=FTYPE)
            else
                object = repeat(block_reduce(readfits(objectfile, FTYPE=FTYPE), dim), 1, 1, nλ)
            end

            for w=1:nλ
                object[:, :, w] .*= spectrum[w]
            end
            object ./= sum(object)
            object .*= flux / Δλ
            return new{FTYPE}(λ, nλ, height, fov, sampling_arcsecperpix, spectrum, flux, background_flux, object)
        else
            if verb == true
                print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Object\n"); print(Crayon(reset=true))
                println("\tSize: $(dim)x$(dim) pixels")
                println("\tFOV: ($(fov)×$(fov)) arcsec")
                println("\tHeight: $(height) km")
                println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
                println("\tNumber of wavelengths: $(length(λ))")
            end
            return new{FTYPE}(λ, nλ, height, fov, sampling_arcsecperpix, spectrum)
        end
    end
end

function mag2flux(mag; D=3.6, ζ=0.0, exptime=20e-3, qe=0.7, adc_gain = 1.0, filter_="V")
    area = pi*(D/2)^2
    airmass = sec(ζ*pi/180)
    nphotons, extinct_coeff = magnitude_zeropoint(filter_=filter_);
    # calculate number of ADU counts on the CCD from star, total
    adu = adc_gain * qe * exptime * area * 10.0^(-0.4*mag) * nphotons * 10^(-0.4*airmass*extinct_coeff);
    return adu 
end

function mag2flux(λ, spectrum, mag, filter_; D=3.6, ζ=0.0, exptime=20e-3)
    ## Flux at top of atmosphere
    area = pi*(D/2)^2
    airmass = sec(ζ*pi/180)
    λfilter, filter_response = filter_.λ, filter_.response
    nphotons_vega = magnitude_zeropoint(λfilter, filter_response);
    nphotons = nphotons_vega * 10^(-(mag+0.3*airmass)/2.5)
    scaled_spectrum = (spectrum ./ sum(spectrum)) .* nphotons
    # calculate number of ADU counts on the CCD from star, per wavelength
    adu_λ = (exptime * area) .* scaled_spectrum
    # calculate number of ADU counts on the CCD from star, total
    adu = length(λ) > 1 ? NumericalIntegration.integrate(λ, adu_λ) : adu_λ[1]
    return adu 
end

function magnitude_zeropoint(; filter_="none")
# Photons per square meter per second produced by a 0th mag star above the atmosphere.
# Assuming spectrum like Vega
    nphot = -1.0;
    coeff = 1.0; # extinction
    if (filter_ == "none") 
        nphot = 4.32e+10;
        coeff = 0.20;
    elseif (filter_ == "U")
        nphot = 5.50e+9;
        coeff = 0.60;
    elseif (filter_ == "B")
        nphot = 3.91e+9;
        coeff = 0.40;
    elseif (filter_ == "V")
        nphot = 8.66e+9;
        coeff = 0.20;
    elseif (filter_ == "R")
        nphot = 1.10e+10;
        coeff = 0.10;
    elseif (filter_ == "I")
        nphot = 6.75e+9;
        coeff = 0.08;
    end
    return nphot, coeff;
end

function magnitude_zeropoint(λmin, λmax, λfilter, filter_response)
    # Photons per square meter per second produced by a 0th mag star above the atmosphere.
    # Assuming spectrum like Vega
    λ = range(λmin, stop=λmax, length=101)
    λ, flux = vega_spectrum(λ=λ)
    filter_itp = interpolate((λfilter,), filter_response, Gridded(Linear()))
    nphot = NumericalIntegration.integrate(λ, flux .* filter_itp(λ))
    return nphot
end

function magnitude_zeropoint(λfilter, filter_response)
    # Photons per square meter per second produced by a 0th mag star above the atmosphere.
    # Assuming spectrum like Vega
    ~, flux = vega_spectrum(λ=λfilter)
    nphot = NumericalIntegration.integrate(λfilter, flux .* filter_response)
    return nphot
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
    spectra = hcat([[sum([materials[i][ll+1] * λ[k].^ll for ll in 0:length(materials[i])-1]) for k = 1:nλ] for i=1:nmaterials]...) ./ 100

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
            for nn=1:10
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
