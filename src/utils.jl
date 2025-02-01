using JLD2
using FITSIO
using Statistics
using DelimitedFiles
using ZernikePolynomials
import Interpolations: LinearInterpolation, Line


function gettype(T)
    return typeof(T).parameters[1]
end

function gettypes(T)
    return typeof(T).parameters
end

function writefits(x, filename; verb=true, header=nothing)
    f = FITS(filename, "w")
    if header !== nothing
        header = FITSHeader(header...)
    end
    write(f, x, header=header)
    close(f)
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
    if λ != []
        itp = LinearInterpolation(λraw, qe_raw, extrapolation_bc=Line())
        qe = itp(λ)
    else
        λ, qe = λraw, qe_raw
    end

    return λ, qe
end

@views function readimages(file::String; FTYPE=Float64)
    images = readfits(file, FTYPE=FTYPE)
    images = repeat(images, 1, 1, 1, 1)
    nsubaps = size(images, 3);
    nepochs = size(images, 4);
    dim = size(images, 1);
    return images, nsubaps, nepochs, dim
end

function calculate_entropy(x)
    p = x ./ sum(x);
    ix = findall(x .> 0);
    return -sum(p[ix] .* log2.(p[ix]));
end

function readmasks(file::String; FTYPE=Float64)
    masks = readfits(file, FTYPE=FTYPE)
    hdu = FITS(file)[1]
    λstart = read_key(hdu, "WAVELENGTH_START")[1]
    λend = read_key(hdu, "WAVELENGTH_END")[1]
    nλ = read_key(hdu, "WAVELENGTH_STEPS")[1]
    λ = FTYPE.(collect(range(λstart, stop=λend, length=nλ)))
    return masks, λ
end

function readmasks(files::Vector{String}; FTYPE=Float64)
    nfiles = length(files)
    masks = Array{Array{FTYPE}}(undef, nfiles)
    for i=1:nfiles
        masks[i] = readfits(files[i], FTYPE=FTYPE)
    end

    hdu = FITS(files[1])[1]
    λstart = read_key(hdu, "WAVELENGTH_START")[1]
    λend = read_key(hdu, "WAVELENGTH_END")[1]
    nλ = read_key(hdu, "WAVELENGTH_STEPS")[1]
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
    file = "data/alpha_lyr_stis_011.fits"
    λ, flux = readspectrum(file, λ=λ)
    return λ, flux
end

function solar_spectrum(; λ=[])
    file = "data/sun_reference_stis_002.fits"
    λ, flux = readspectrum(file, λ=λ)
    return λ, flux
end

function readspectrum(file; λ=[])
    h = 6.626196e-27  # erg⋅s
    c = 2.997924562e17  # nm/s

    λ₀ = read(FITS(file)[2], "WAVELENGTH")  # Å
    λ₀ ./= 10  # convert to nm
    flux = read(FITS(file)[2], "FLUX")  # erg/s/cm^2/Å
    flux ./= h*c ./ λ₀  # convert erg/s to ph/s [ph/s/cm^2/Å]
    flux .*= 1e4  # convert cm^2 to m^2 [ph/s/m^2/Å]
    flux .*= 10  # convert Å to nm [ph/s/m^2/nm]

    if λ == []
        λ = λ₀
    else
        flux = interpolate1d(λ₀, flux, λ)
    end

    return λ, flux
end

function ft(x)
    return ifftshift(fft(ifftshft(x)))
end

function ift(x)
    return fftshift(ifft(ifftshift(x)))
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
        # fftshift!(container, in)
        # mul!(container, pft, container)
        # ifftshift!(out, container)
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
        # ifftshift!(container, in)
        # mul!(container, pift, container)
        # fftshift!(out, container)
        mul!(out, pift, in)
        out .*= scale_ifft
    end

    return ifft!, container
end

function setup_conv(::Type{T}, dim) where {T}
    scale_conv = T(dim)
    ft1, container1 = setup_fft(T, dim)
    ft2, container2 = setup_fft(T, dim)
    ift1, ~ = setup_ifft(T, dim)
    function conv!(out, in1, in2)
        ft1(container1, in1)
        ft2(container2, in2)
        container1 .*= container2
        ift1(out, container1)
        out ./= scale_conv
    end

    return conv!
end

function preconvolve(in)
    dim = size(in, 1)
    FTYPE = eltype(in)
    scale_conv = FTYPE(dim)
    ft1, container1 = setup_fft(FTYPE, dim)
    ft2, container2 = setup_fft(FTYPE, dim)
    ift1, ~ = setup_ifft(FTYPE, dim)
    ft2(container2, in)
    function preconv!(out, in)
        ft1(container1, in)
        container1 .*= container2
        ift1(out, container1)
        out ./= scale_conv
    end

    return preconv!
end

function setup_corr(::Type{T}, dim) where {T}
    scale_corr = T(dim)
    ft1, container1 = setup_fft(T, dim)
    ft2, container2 = setup_fft(T, dim)
    ift1, ~ = setup_ifft(T, dim)
    function corr!(out, in1, in2)
        ft1(container1, in1)
        ft2(container2, in2)
        container1 .*= conj.(container2)
        ift1(out, container1)
        out ./= scale_corr
    end

    return corr!
end

function precorrelate(in)
    dim = size(in, 1)
    FTYPE = eltype(in)
    scale_corr = FTYPE(dim)
    ft1, container1 = setup_fft(FTYPE, dim)
    ft2, container2 = setup_fft(FTYPE, dim)
    ift1, ~ = setup_ifft(FTYPE, dim)
    ft2(container2, in)
    conj!(container2)
    function precorr!(out, in)
        ft1(container1, in)
        container1 .*= container2
        ift1(out, container1)
        out ./= scale_corr
    end
    
    return precorr!
end

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

@views function stack2mosaic(stack::AbstractVector{<:AbstractFloat}, nside, ix)
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

@views function stack2mosaic(stack::AbstractArray{<:AbstractFloat, 3}, nside, ix)
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
    F_smooth = conv_psf(F, k)
    return F_smooth
end

function interpolate1d(x₀, y₀, x)
    itp = LinearInterpolation(x₀, y₀, extrapolation_bc=Line())
    y = itp(x)
    return y
end

function readtransmission(filename; resolution=Inf, λ=[])
    λtransmission, transmission = readfile(filename)
    if resolution != Inf
        transmission .= smooth_to_resolution(λtransmission, transmission, resolution)
    end

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
