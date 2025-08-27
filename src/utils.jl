using JLD2
using FFTW
using FITSIO
using Statistics
using DelimitedFiles
import Interpolations: LinearInterpolation, Line
FFTW.set_provider!("fftw")
FFTW.set_num_threads(1)


mutable struct ConvolutionPlan{T<:Number}
    scale::T
    ft1::Union{FFTW.rFFTWPlan{T, -1, false, 2, Tuple{Int64, Int64}}, FFTW.cFFTWPlan{Complex{T}, -1, false, 2, UnitRange{Int64}}}
    ft2::Union{FFTW.rFFTWPlan{T, -1, false, 2, Tuple{Int64, Int64}}, FFTW.cFFTWPlan{Complex{T}, -1, false, 2, UnitRange{Int64}}}
    ift1::Union{AbstractFFTs.ScaledPlan{Complex{T}, FFTW.cFFTWPlan{Complex{T}, 1, false, 2, UnitRange{Int64}}, T}, AbstractFFTs.ScaledPlan{Complex{T}, FFTW.rFFTWPlan{Complex{T}, 1, false, 2, UnitRange{Int64}}, T}}
    container1::Matrix{Complex{T}}
    container2::Matrix{Complex{T}}
    function ConvolutionPlan(dim; FTYPE=Float64)
        scale_conv = FTYPE(dim)
        ft1, container1 = select_fft_plan(FTYPE, dim)
        ft2, container2 = select_fft_plan(FTYPE, dim)
        ift1, ~ = select_ifft_plan(FTYPE, dim)
        return new{FTYPE}(scale_conv, ft1, ft2, ift1, container1, container2)
    end
end

function convolve!(out::AbstractMatrix{T}, plan::ConvolutionPlan{T}, in1::AbstractMatrix{T}, in2::AbstractMatrix{T}) where {T<:Number}
    mul!(plan.container1, plan.ft1, in1)
    mul!(plan.container2, plan.ft2, in2)
    plan.container2 .*= plan.container1
    mul!(out, plan.ift1, plan.container2)
end

mutable struct Preconvolve{T<:Number}
    kernel::Matrix{T}
    plan::ConvolutionPlan{T}
    function Preconvolve(kernel)
        dim = size(kernel, 1)
        FTYPE = eltype(kernel)
        plan = ConvolutionPlan(dim, FTYPE=FTYPE)
        mul!(plan.container1, plan.ft1, kernel)
        return new{FTYPE}(kernel, plan)
    end
end

function convolve!(out::AbstractMatrix{T}, preconv::Preconvolve{T}, in::AbstractMatrix{T}) where {T<:Number}
    mul!(preconv.plan.container2, preconv.plan.ft2, in)
    preconv.plan.container2 .*= preconv.plan.container1
    mul!(out, preconv.plan.ift1, preconv.plan.container2)
end

mutable struct CorrelationPlan{T<:Number}
    scale::T
    ft1::Union{FFTW.rFFTWPlan{T, -1, false, 2, Tuple{Int64, Int64}}, FFTW.cFFTWPlan{Complex{T}, -1, false, 2, UnitRange{Int64}}}
    ft2::Union{FFTW.rFFTWPlan{T, -1, false, 2, Tuple{Int64, Int64}}, FFTW.cFFTWPlan{Complex{T}, -1, false, 2, UnitRange{Int64}}}
    ift1::Union{AbstractFFTs.ScaledPlan{Complex{T}, FFTW.cFFTWPlan{Complex{T}, 1, false, 2, UnitRange{Int64}}, T}, AbstractFFTs.ScaledPlan{Complex{T}, FFTW.rFFTWPlan{Complex{T}, 1, false, 2, UnitRange{Int64}}, T}}
    container1::Matrix{Complex{T}}
    container2::Matrix{Complex{T}}
    function CorrelationPlan(dim; FTYPE=Float64)
        scale_corr = FTYPE(dim)
        ft1, container1 = select_fft_plan(FTYPE, dim)
        ft2, container2 = select_fft_plan(FTYPE, dim)
        ift1, ~ = select_ifft_plan(FTYPE, dim)
        return new{FTYPE}(scale_corr, ft1, ft2, ift1, container1, container2)
    end
end

function correlate!(out::AbstractMatrix{T}, plan::CorrelationPlan{T}, in1::AbstractMatrix{T}, in2::AbstractMatrix{T}) where {T<:Number}
    mul!(plan.container1, plan.ft1, in1)
    mul!(plan.container2, plan.ft2, in2)
    conj!(plan.container2)
    plan.container1 .*= plan.container2
    mul!(out, plan.ift1, plan.container1)
end

mutable struct Precorrelate{T<:Number}
    kernel::Matrix{T}
    plan::CorrelationPlan{T}
    function Precorrelate(kernel)
        dim = size(kernel, 1)
        FTYPE = eltype(kernel)
        plan = CorrelationPlan(dim, FTYPE=FTYPE)
        mul!(plan.container1, plan.ft1, kernel)
        conj!(plan.container1)
        return new{FTYPE}(kernel, plan)
    end
end

function correlate!(out::AbstractMatrix{T}, precorr::Precorrelate{T}, in::AbstractMatrix{T}) where {T<:Number}
    mul!(precorr.plan.container2, precorr.plan.ft2, in)
    precorr.plan.container2 .*= precorr.plan.container1
    mul!(out, precorr.plan.ift1, precorr.plan.container2)
end

function gettype(T)
    return gettypes(T)[1]
end

<<<<<<< HEAD
function gettypes(T)
    return typeof(T).parameters
end

function create_header(λ::AbstractVector{<:AbstractFloat}; units::String="unitless")
    λmin = minimum(λ)
    λmax = maximum(λ)
    nλ = length(λ)
    header = (
        ["WAVELENGTH_START", "WAVELENGTH_END", "WAVELENGTH_STEPS", "BUNIT"], 
        [λmin, λmax, nλ, units],
        ["Shortest wavelength of mask [m]", "Largest wavelength of mask [m]", "Number of wavelength steps", "Image Units"]
    )
    return header
end

function create_header(observations::Observations{<:Number, <:Number})
    λmin = minimum(observations.detector.λ)
    λmax = maximum(observations.detector.λ)
    nλ = length(observations.detector.λ)
    header = (
        ["WAVELENGTH_START", "WAVELENGTH_END", "WAVELENGTH_STEPS", "BUNIT", "EXPTIME", "TIME-START", "TELAPSE"], 
        [λmin, λmax, nλ, "counts", observations.detector.exptime, minimum(observations.times), observations.detector.exptime],  # Assumes frame time is the same as exposure time
        ["Shortest wavelength of mask [m]", "Largest wavelength of mask [m]", "Number of wavelength steps", "Image Units", "Exposure Time [s]", "Start time of the observations [s]", "Elapsed time of each frame [s]"]
    )
    return header
end

function writefits(x, filename; verb=true, header=nothing, times=nothing)
    FITS(filename, "w") do f
        if !isnothing(header)
            header = FITSHeader(header...)
        end
        write(f, x, header=header)

        if !isnothing(times)
            write(f, times)
        end
=======
function writefits(x, filename; verb=true, header=nothing)
    f = FITS(filename, "w")
    if header !== nothing
        header = FITSHeader(header...)
>>>>>>> main
    end
    
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function writefile(x::Vector{<:AbstractFloat}, filename; header="", verb=true)
    f = open(filename, "w")
    if header != ""
        write(f, header * "\n")
    end
    for i=1:length(x)
        write(f, "$(i)\t$(x[i])\n")
    end
    close(f)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function writefile(x, y, filename; header="", verb=true)
    f = open(filename, "w")
    if header != ""
        write(f, header * "\n")
    end
    for i=1:length(x)
        write(f, "$(x[i])\t$(y[i])\n")
    end
    close(f)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function writeobject(x, filename; verb=true)
    jldsave(filename; x)
    if verb == true
        print("Written to "); printstyled("$(filename)\n", color=:red)
    end
end

function readobject(filename::String)
    return read(jldopen(filename, "r"), "x")
end

function readfile(filename)
    data = readdlm(filename)
    return data[:, 1], data[:, 2]
end

function readqe(filename; λ=[])
    λraw, qe_raw = readfile(filename)
    λraw .*= 1e-9
    if λ != []
        itp = LinearInterpolation(λraw, qe_raw, extrapolation_bc=Line())
        qe = max.(0.0, itp(λ))
    else
        λ, qe = λraw, qe_raw
    end

    return λ, qe
end

@views function readimages(file::String; FTYPE=Float64)
    f = FITS(file)
    hdr = read_header(file)

    images = FTYPE.(read(f[1]))
    images = repeat(images, 1, 1, 1, 1)
    nsubaps = size(images, 3);
    nepochs = size(images, 4);
    dim = size(images, 1);
    
    exptime = hdr["EXPTIME"]
    elapsed = hdr["TELAPSE"]
    tstart = hdr["TIME-START"]
    times = try
        times = read(f[2])
    catch
        times = []
    end

    return images, nsubaps, nepochs, dim, exptime, times
end

function calculate_entropy(x)
    p = x ./ sum(x);
    ix = findall(x .> 0);
    return -sum(p[ix] .* log2.(p[ix]));
end

function readmasks(file::String; FTYPE=Float64)
    masks = readfits(file, FTYPE=FTYPE)
    hdr = read_header(file)
    λstart = hdr["WAVELENGTH_START"]
    λend = hdr["WAVELENGTH_END"]
    nλ = hdr["WAVELENGTH_STEPS"]
    λ = FTYPE.(collect(range(λstart, stop=λend, length=nλ)))
    return masks, λ
end

function readmasks(files::Vector{String}; FTYPE=Float64)
    nfiles = length(files)
    masks = Array{Array{FTYPE}}(undef, nfiles)
    for i=1:nfiles
        masks[i] = readfits(files[i], FTYPE=FTYPE)
    end

    hdr = read_header(files[1])
    λstart = hdr["WAVELENGTH_START"]
    λend = hdr["WAVELENGTH_END"]
    nλ = hdr["WAVELENGTH_STEPS"]
    λ = FTYPE.(collect(range(λstart, stop=λend, length=nλ)))

    masks = cat(masks..., dims=3)
    return masks, λ
end

function readfits(file; FTYPE=Float64)
    return FTYPE.(read(FITS(file)[1]))
end

function gaussian_kernel(dim, fwhm; FTYPE=Float64)
    coord = (1:dim) .- (dim÷2 + 1)
    rr = hypot.(coord, coord')
    σ = fwhm / 2.35482
    k = exp.(-rr.^2 ./ σ^2)
    k ./= sum(k)
    return FTYPE.(k)
end

@views function shift_and_add(images, nsubaps, nepochs, dim; FTYPE=Float64)
    image_sum = zeros(FTYPE, dim, dim)
    kernel = zeros(FTYPE, dim, dim)
    for n=1:nsubaps
        for t=1:nepochs
            fill!(kernel, zero(FTYPE))
            center = Tuple(argmax(images[:, :, n, t]))
            Δy, Δx = (dim÷2)+1 - center[1], (dim÷2)+1 - center[2]
            shift = CartesianIndex((dim÷2+1 + Δy, dim÷2+1 + Δx))
            kernel[shift] = 1.0
            image_sum .+= conv_psf(images[:, :, n, t], kernel)
        end
    end

    return FTYPE.(image_sum ./ (nsubaps*nepochs))
end

function block_reduce(image, newdim)
    FTYPE = eltype(image)
    dim = size(image, 1)
    if newdim != dim
        newimage = zeros(FTYPE, newdim, newdim)
        block_reduce!(newimage, image)
    else
        newimage = image
    end
    
    return newimage
end

@views function block_reduce!(newimage, image)
    dim = size(image, 1)
    newdim = size(newimage, 1)
    pixperbin = dim ÷ newdim
    for i=1:newdim
        for j=1:newdim
            newimage[i, j] = sum(image[(i-1)*pixperbin + 1:i*pixperbin, (j-1)*pixperbin + 1:j*pixperbin])
        end
    end
end

function block_replicate(image, newdim)
    FTYPE = eltype(image)
    dim = size(image, 1)
    if newdim != dim
        newimage = zeros(FTYPE, newdim, newdim)
        block_replicate!(newimage, image)
    else
        newimage = image
    end
    return newimage
end

@views function block_replicate!(newimage, image)
    dim = size(image, 1)
    newdim = size(newimage, 1)
    pixperbin = newdim ÷ dim
    for i=1:dim
        for j=1:dim
            newimage[(i-1)*pixperbin + 1:i*pixperbin, (j-1)*pixperbin + 1:j*pixperbin] .= image[i, j] / pixperbin^2
        end
    end
end

function fit_plane(ϕ, mask)
    dim = size(ϕ, 1)
    ix = (mask .> 0)
    X = (collect(1:dim)' .* ones(dim))
    Y = X'
    Z = ϕ[ix]
    N = length(X[ix])
    M = [X[ix] Y[ix] ones(N)]

    fit = inv(M' * M) * M' * Z
    ll = sqrt( fit[1]^2 + fit[2]^2 + 1 )
    a, b, c, d = -fit[1]/ll, -fit[2] / ll, 1 / ll, -fit[3] / ll

    plane = -((a/c .* X) .+ (b/c .* Y) .+ (d/c))
    return plane
end

function crop(x, newdim)
    dim = size(x, 1)
    if dim == newdim
        return x
    else
        ix1 = dim÷2-newdim÷2
        ix2 = dim÷2+newdim÷2-1
        return x[ix1:ix2, ix1:ix2]
    end
end

function smooth_to_rmse!(ϕ_smooth, ϕ, rms_target, mask, dim; FTYPE=Float64)
    rms = Inf
    fwhm = dim/10
    while (abs(rms-rms_target) > 1e-3)
        k = gaussian_kernel(dim, fwhm, FTYPE=FTYPE)
        ϕ_smooth .= conv_psf(ϕ, k)
        rms = sqrt(mean((ϕ_smooth[mask .> 0] .- ϕ[mask .> 0]).^2))
        # println(fwhm, " ", rms)
        if sign(rms - rms_target) == 1 
            fwhm -= fwhm*0.005
        elseif sign(rms - rms_target) == -1
            fwhm += fwhm*0.005
        end
    end
end

function bartlett_hann1d(n, N)
    return 0.62 .- 0.48 .* abs.(n ./ N .- 0.5) .- 0.38 .* cos.(2pi .* n ./ N)
end

function bartlett_hann1d_centered(n, N)
    nn = n .+ (N+1 ÷ 2)
    return bartlett_hann1d(nn, N)
end

function bartlett_hann2d(i, j, N)
    return bartlett_hann1d(i, N) .* bartlett_hann1d(j, N)
end

function super_gaussian(dim, σ, n; FTYPE=Float64)
    x = (1:dim) .- (dim÷2+1)
    r = hypot.(x, x')
    sg = exp.(-(r ./ σ).^n)
    return FTYPE.(sg)
end

function vega_spectrum(; λ=[])
<<<<<<< HEAD
    file = "$(@__DIR__)/../data/alpha_lyr_stis_011.fits"
    λ₀, spectral_irradiance = readspectrum(file)  # erg/s/cm^2/Å
    λ₀ .*= 1e-10  # convert Å to m
    spectral_irradiance .*= 1e-7  # convert erg/s to W [W/cm^2/Å]
    spectral_irradiance .*= 1e4  # convert cm^2 to m^2 [W/m^2/Å]
    spectral_irradiance .*= 1e10  # convert Å to m [W/m^2/m]
=======
    file = "data/alpha_lyr_stis_011.fits"
    λ, flux = read_spectrum(file, λ=λ)
    return λ, flux
end

function solar_spectrum(; λ=[])
    file = "data/sun_reference_stis_002.fits"
    λ, flux = read_spectrum(file, λ=λ)
    return λ, flux
end

function read_spectrum(file; λ=[])
    h = 6.626196e-27  # erg⋅s
    c = 2.997924562e17  # nm/s

    λ₀ = read(FITS(file)[2], "WAVELENGTH")  # Å
    λ₀ ./= 10  # convert to nm
    flux = read(FITS(file)[2], "FLUX")  # erg/s/cm^2/Å
    flux ./= h*c ./ λ₀  # convert erg/s to ph/s [ph/s/cm^2/Å]
    flux .*= 1e4  # convert cm^2 to m^2 [ph/s/m^2/Å]
    flux .*= 10  # convert Å to nm [ph/s/m^2/nm]

>>>>>>> main
    if λ == []
        return λ₀, flux
    else
<<<<<<< HEAD
        spectral_irradiance = interpolate1d(λ₀, spectral_irradiance, λ)
    end
    return λ, spectral_irradiance
end

function solar_spectrum(; λ=[])
    file = "$(@__DIR__)/../data/sun_reference_stis_002.fits"
    λ₀, spectral_irradiance = readspectrum(file)  # Å, mW/m^2/Å
    λ₀ .*= 1e-10  # convert Å to m
    spectral_irradiance .*= 1e-3  # convert mW to W [W/m^2/Å]
    spectral_irradiance .*= 1e10  # convert Å to m [W/m^2/m]
    if λ == []
        λ = λ₀
    else
        spectral_irradiance = interpolate1d(λ₀, spectral_irradiance, λ)
    end

    return λ, spectral_irradiance
end

function readspectrum(file)
    λ₀ = read(FITS(file)[2], "WAVELENGTH")
    spectral_irradiance = read(FITS(file)[2], "FLUX")
    return λ₀, spectral_irradiance
end

function ft(x)
    return fftshift(fft(ifftshift(x)))
end

function ift(x)
    return fftshift(ifft(ifftshift(x)))
=======
        itp = interpolate((λ₀,), flux, Gridded(Linear()))
        flux = itp(λ)
        return λ, flux
    end
>>>>>>> main
end

function fourier_filter(x, r; FTYPE=Float64)
    ## Keep frequencies inside r
    dim = size(x, 1)
    mask = ones(FTYPE, dim, dim)
    xx = (1:dim) .* ones(dim)'
    xx .-= dim÷2+1
    rr = hypot.(xx, xx')
    mask[rr .> r] .= 0

    X = ft(x)
    x_filtered = real.(ift(X .* mask))
    return x_filtered
end

function filter_to_rmse!(ϕ_smooth, ϕ, rms_target, mask, dim; FTYPE=Float64)
    samples = dim-dim÷4:dim
    rms = zeros(FTYPE, length(samples))
    i = 1
    for n in samples
        ϕ_smooth .= fourier_filter(ϕ, dim-n)
        rms[i] = sqrt(mean((ϕ_smooth[mask .> 0] .- ϕ[mask .> 0]).^2))
        i += 1
    end
    ϕ_smooth .= fourier_filter(ϕ, dim-argmin(abs.(rms .- rms_target)))
end

function select_fft_plan(::Type{T}, dim) where {T<:Real}
    return plan_rfft(Matrix{T}(undef, dim, dim)), Matrix{Complex{T}}(undef, dim÷2+1, dim)
end

function select_fft_plan(::Type{T}, dim) where {T<:Complex}
    return plan_fft(Matrix{T}(undef, dim, dim)), Matrix{T}(undef, dim, dim)
end

function setup_fft(::Type{T}, dim) where {T}
    pft, container = select_fft_plan(T, dim)
    function fft!(out, in)
        mul!(out, pft, in)
    end

    return fft!, container
end

function select_ifft_plan(::Type{T}, dim) where {T<:Real}
    return plan_irfft(Matrix{Complex{T}}(undef, dim÷2+1, dim), dim), Matrix{T}(undef, dim, dim)
end

function select_ifft_plan(::Type{T}, dim) where {T<:Complex}
    return plan_ifft(Matrix{T}(undef, dim, dim)), Matrix{T}(undef, dim, dim)
end

function setup_ifft(::Type{T}, dim) where {T}
    scale_ifft = T(dim)
    pift, container = select_ifft_plan(T, dim)
    function ifft!(out, in)
        mul!(out, pift, in)
        out .*= scale_ifft
    end

    return ifft!, container
end

<<<<<<< HEAD
function setup_autocorr(::Type{T}, dim) where {T<:Real}
    container = Matrix{Complex{T}}(undef, dim, dim)
    ft1, ~ = setup_fft(Complex{T}, dim)
    ift1, container2 = setup_ifft(Complex{T}, dim)
    function autocorr!(out, in)
        container .= complex.(in)
        ift1(container2, container)
        container .= abs2.(container2)
        ft1(container2, container)
        out .= real.(container2)
    end

    return autocorr!
end

function setup_autocorr(::Type{T}, dim) where {T<:Complex}
    ft1, container1 = setup_fft(T, dim)
    ift1, container2 = setup_ifft(T, dim)
    function autocorr!(out, in)
        ift1(container2, in)
        container1 .= abs2.(container2)
        ft1(container2, container1)
        out .= container2
=======
function setup_conv(dim; FTYPE=Float64)
    ft1 = setup_fft(dim, FTYPE=FTYPE)
    ft2 = setup_fft(dim, FTYPE=FTYPE)
    ift1 = setup_ifft(dim, FTYPE=FTYPE)
    container1 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container2 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container3 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    function conv!(out, in1, in2)
        ft1(container1, in1)
        ft2(container2, in2)
        container3 .= container1 .* container2
        ift1(out, container3)
    end

    return conv!
end

function setup_corr(dim; FTYPE=Float64)
    ft1 = setup_fft(dim, FTYPE=FTYPE)
    ft2 = setup_fft(dim, FTYPE=FTYPE)
    ift1 = setup_ifft(dim, FTYPE=FTYPE)
    container1 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container2 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    container3 = Matrix{Complex{FTYPE}}(undef, dim, dim)
    function corr!(out, in1, in2)
        ft1(container1, in1)
        ft2(container2, in2)
        container3 .= container1 .* conj.(container2)
        ift1(out, container3)
    end

    return corr!
end

function setup_autocorr(dim; FTYPE=Float64)
    ift1 = setup_ifft(dim, FTYPE=FTYPE)
    container = Matrix{Complex{FTYPE}}(undef, dim, dim)
    function autocorr!(out, in)
        ift1(container, in)
        out .= abs2.(container)
>>>>>>> main
    end

    return autocorr!
end

function setup_operator_mul(dim; FTYPE=Float64)
    container = Matrix{FTYPE}(undef, dim, dim)
    function apply!(in, operator)
        mul!(container, operator, in)
        return container
    end

    return apply!
end

function calculate_ssim(x, y)
    X = x; Y = y
    # X = x .+ minimum(x); Y = y .+ minimum(y)
    maxVal = max(maximum(X), maximum(Y))
    X ./= maxVal; Y ./= maxVal

    μx = mean(X); μy = mean(Y)
    σx = std(X); σy = std(Y)
    σxy = cov(vec(X), vec(Y))
    
    k1 = 0.01; k2 = 0.03
    
    l = (2*μx*μy + k1^2) / (μx^2 + μy^2 + k1^2)
    c = (2*σx*σy + k2^2) / (σx^2 + σy^2 + k2^2)
    s = (σxy + k2^2/4) / (σx*σy + k2^2/4)
    ssim = l * c * s
    return ssim
end

@views function stack2mosaic(stack::AbstractVector{<:Number}, nside, ix)
    ## 1d vec of numbers to 2d image
    FTYPE = eltype(stack)
    mosaic = zeros(FTYPE, nside, nside)
    n1 = n2 = 1
    for y=1:nside
        for x=1:nside
            if (n1 in ix) == true
                mosaic[y, x] = stack[n2]
                n2 += 1
            end
            n1 += 1     
        end
    end
 
    return mosaic
end

<<<<<<< HEAD
@views function stack2mosaic(stack::AbstractArray{<:Number, 3}, nside, ix)
=======
function stack2mosaic(stack::AbstractArray{<:AbstractFloat, 3}, nside, ix)
>>>>>>> main
    ## 3d stack of images to 2d mosaic image
    FTYPE = eltype(stack)
    dim = size(stack, 1)
    mosaic = zeros(FTYPE, dim*nside, dim*nside)
    n1 = 1
    n2 = 1
    for y=1:nside
        ixy = (y-1)*dim+1:y*dim
        for x=1:nside
            ixx = (x-1)*dim+1:x*dim
            if (n1 in ix) == true
                mosaic[ixy, ixx] .= stack[:, :, n2]
                n2 += 1
            end
            n1 += 1     
        end
    end

    return mosaic
end
<<<<<<< HEAD

function create_zernike_screen(dim, radius, index, waves; FTYPE=Float64)
    x = collect(-dim÷2:dim÷2-1) ./ radius
    ϕz = evaluatezernike(x, x, [Noll(index)], [waves])
    N = normalization(Noll(index))
    ϕz .*= pi / N
    return FTYPE.(ϕz)
end

function smooth_to_resolution(λ, F, resolution)
    nλ = length(λ)
    x = (1:nλ) .- (nλ÷2+1)
    Δλ = mean(λ) / resolution
    Δλ₀ = (maximum(λ) - minimum(λ)) / (nλ - 1)
    fwhm = Δλ / Δλ₀
    σ = fwhm / 2.35482
    k = exp.(-x.^2 ./ σ^2)
    k ./= sum(k)
    F_smooth = conv(F, k)
    return F_smooth
end

function interpolate1d(x₀, y₀, x)
    itp = LinearInterpolation(x₀, y₀, extrapolation_bc=Line())
    y = itp(x)
    return y
end

function readtransmission(filename; resolution=Inf, λ=[])
    λtransmission, transmission = readfile(filename)
    λtransmission .*= 1e-9
    # if resolution != Inf
    #     transmission .= smooth_to_resolution(λtransmission, transmission, resolution)
    # end

    if λ == []
        λ = λtransmission
    else
        transmission = interpolate1d(λtransmission, transmission, λ)
    end

    return λ, transmission
end

function center_of_gravity(image)
    dim = size(image, 1)
    xx = ((1:dim) .- (dim÷2+1))' .* ones(dim)
    yy = xx'
    Δx = sum(xx .* image) / sum(image)
    Δy = sum(yy .* image) / sum(image)
    return Δx, Δy
end

function zeros!(A::AbstractArray{T}) where {T<:Number}
    fill!(A, zero(eltype(A)))
end

function ones!(A::AbstractArray{T}) where {T<:Number}
    fill!(A, one(eltype(A)))
end
=======
>>>>>>> main
