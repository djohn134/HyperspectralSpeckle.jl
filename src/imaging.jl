using Distributions
using ProgressMeter
using TwoDimensional
using LinearInterpolators

"""
    convert_image(out, in)

When both `out` and `in` are floating-point matrices, 
`convert_image` just copies `in` into `out`. We do not
take the `floor` of `in` here because the `floor` function
is not differentiable. Thus, this function should be used
when reconstructing images.
"""
function convert_image(out::AbstractMatrix{<:AbstractFloat}, in::AbstractMatrix{<:AbstractFloat})
    out .= in
end

"""
    convert_image(out, in)

When `out` is an integer-type matrix, `convert_image` takes 
the `floor` of `in`, converting to the integer type of `out`, 
then copies the result to `out`. This behavior mimics the 
quantization of pixel counts by a detector and should be
used when synthesizing images.
"""
function convert_image(out::AbstractMatrix{<:Integer}, in::AbstractMatrix{<:AbstractFloat})
    out .= floor.(eltype(out), in)
end

function create_extractor_operator(position, screen_dim, output_dim, scaleby_height, scaleby_wavelength; FTYPE=FTYPE)
    kernel = LinearSpline(FTYPE)
    transform = AffineTransform2D{FTYPE}()
    screen_size = (Int64(screen_dim), Int64(screen_dim))
    output_size = (Int64(output_dim), Int64(output_dim))
    full_transform = ((transform + Tuple(position)) * (1/(scaleby_height*scaleby_wavelength))) - (output_dim÷2, output_dim÷2)
    extractor = TwoDimensionalTransformInterpolator(output_size, screen_size, kernel, full_transform)
    return extractor
end

function create_extractor_adjoint(position, screen_dim, input_dim, scaleby_height, scaleby_wavelength; FTYPE=FTYPE)
    kernel = LinearSpline(FTYPE)
    transform = AffineTransform2D{FTYPE}()
    screen_size = (Int64(screen_dim), Int64(screen_dim))
    input_size = (Int64(input_dim), Int64(input_dim))
    full_transform = ((transform + (input_dim÷2, input_dim÷2)) * (scaleby_height*scaleby_wavelength)) - Tuple(position)
    extractor_adj = TwoDimensionalTransformInterpolator(screen_size, input_size, kernel, full_transform)
    return extractor_adj
end

function position2phase!(ϕ_out, ϕ_full, extractor)
    mul!(ϕ_out, extractor, ϕ_full)
end

function create_refraction_operator(λ, λ_ref, ζ, pixscale, build_dim; FTYPE=Float64)
    kernel = LinearSpline(FTYPE)
    transform = AffineTransform2D{FTYPE}()
    θref = get_refraction(λ_ref, ζ)
    θλ = get_refraction(λ, ζ)
    build_size = (Int64(build_dim), Int64(build_dim))
    Δpix = FTYPE((θλ-θref)*206265 / pixscale)
    refraction = TwoDimensionalTransformInterpolator(build_size, build_size, kernel, transform - (Δpix, 0))
    return refraction
end

function create_refraction_adjoint(λ, λ_ref, ζ, pixscale, build_dim; FTYPE=Float64)
    kernel = LinearSpline(FTYPE)
    transform = AffineTransform2D{FTYPE}()
    θref = get_refraction(λ_ref, ζ)
    θλ = get_refraction(λ, ζ)
    build_size = (Int64(build_dim), Int64(build_dim))
    Δpix = FTYPE((θλ-θref)*206265 / pixscale)
    refraction = TwoDimensionalTransformInterpolator(build_size, build_size, kernel, transform + (Δpix, 0))
    return refraction
end

function pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, scale_psf, ifft_prealloc!::Function, refraction)
    P .= mask .* scale_psf .* A .* cis.(ϕ)
    ifft_prealloc!(p, P)
    psf_temp .= abs2.(p)
    fftshift!(psf, psf_temp)
    mul!(psf_temp, refraction, psf)
    fftshift!(psf, psf_temp)
end

function add_noise!(image, rn, poisson::Bool; FTYPE=Float64)
    if poisson == true
        image .= FTYPE.(rand.(Distributions.Poisson.(image)))
    end
    ## Read noise has a non-zero mean and sigma!
    image .+= rn .* randn(FTYPE, size(image))
    image .= max.(image, Ref(zero(FTYPE)))
end

@views function calculate_composite_pupil!(A, ϕ_composite, ϕ_slices, ϕ_full, nlayers, extractors, mask, sampling_nyquist_mperpix, heights; propagate=true)
    calculate_composite_phase!(ϕ_composite, ϕ_slices, ϕ_full, nlayers, extractors)
    # if propagate == true
    #     calculate_composite_amplitude!(A, mask, nlayers, ϕ_slices, sampling_nyquist_mperpix, heights)
    # end
end

@views function calculate_composite_phase!(ϕ_composite, ϕ_slices, ϕ_full, nlayers, extractors)
    fill!(ϕ_composite, zero(eltype(ϕ_composite)))
    for l=1:nlayers
        position2phase!(ϕ_slices, ϕ_full[:, :, l], extractors[l])
        ϕ_composite .+= ϕ_slices
    end
end

@views function calculate_composite_amplitude!(A, mask, nlayers, ϕ_slices, sampling_nyquist_mperpix, heights)
    FTYPE = eltype(A)
    ϕ_slices_subap = repeat(mask, 1, 1, nlayers) .* ϕ_slices
    N = size(ϕ_slices, 1)
    x1 = ((-N÷2:N÷2-1) .* sampling_nyquist_mperpix[1])' .* ones(N)
    y1 = x1'
    sg = repeat(exp.(-(x1 ./ (0.47*N)).^16) .* exp.(-(y1 ./ (0.47*N)).^16), 1, 1, nlayers)
    Uout = propagate_layers(ones(Complex{FTYPE}, size(mask)), λ, sampling_nyquist_mperpix[1], sampling_nyquist_mperpix[end], heights, sg .* cis.(ϕ_slices_subap))
    A .= mask .* abs.(Uout)
end

@views function create_detector_images(patches, observations, atmosphere, object; build_dim=object.dim, noise=false, verb=true)
    FTYPE = gettype(patches)
    nthreads = Threads.nthreads()
    psfs = zeros(FTYPE, build_dim, build_dim, nthreads)
    psf_temp = zeros(FTYPE, build_dim, build_dim, nthreads)
    image_big_temp = zeros(FTYPE, build_dim, build_dim, nthreads)
    object_patch = zeros(FTYPE, build_dim, build_dim, nthreads)
    P = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    p = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    A = ones(FTYPE, build_dim, build_dim, nthreads)
    ϕ_slices = zeros(FTYPE, build_dim, build_dim, nthreads)
    ϕ_composite = zeros(FTYPE, build_dim, build_dim, nthreads)
    iffts = [setup_ifft(Complex{FTYPE}, build_dim)[1] for tid=1:Threads.nthreads()]
    convs = [setup_conv(FTYPE, build_dim) for tid=1:Threads.nthreads()]

    scaleby_height = layer_scale_factors(atmosphere.heights, object.range)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    if verb == true
        println("Creating $(observations.dim)×$(observations.dim) images for $(observations.nepochs) times and $(observations.nsubaps) subaps")
    end
    DTYPE = gettypes(observations.detector)[end]
    image_small_temp = zeros(FTYPE, observations.dim, observations.dim, nthreads)
    refraction = [create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations.ζ, observations.detector.pixscale, build_dim, FTYPE=FTYPE) for w=1:atmosphere.nλ]
    extractors = create_patch_extractors(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height, build_dim=build_dim)    
    images_float = zeros(FTYPE, observations.dim, observations.dim, nthreads)
    observations.images = zeros(DTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs)
    create_detector_images!(patches, observations, atmosphere, object, images_float, image_small_temp, image_big_temp, psfs, psf_temp, object_patch, A, P, p, refraction, iffts, convs, ϕ_composite, ϕ_slices, extractors, noise=noise)
end

@views function create_detector_images!(patches, observations, atmosphere, object, images_float, image_small_temp, image_big_temp, psf, psf_temp, object_patch, A, P, p, refraction, iffts, convs, ϕ_composite, ϕ_slices, extractors; noise=false)
    FTYPE = gettype(observations)
    nλint = 1
    smoothing!(out, in) = nothing
    prog = Progress(observations.nepochs*observations.nsubexp*observations.nsubaps)
    Threads.@threads :static for t=1:observations.nepochs
        tid = Threads.threadid()
        for n=1:observations.nsubaps
            fill!(images_float[:, :, tid], zero(FTYPE))
            for tsub=1:observations.nsubexp 
                create_radiant_energy_pre_detector!(images_float[:, :, tid], image_small_temp[:, :, tid], image_big_temp[:, :, tid], psf[:, :, tid], psf_temp[:, :, tid], observations.masks.scale_psfs, object.object, patches.w, object_patch[:, :, tid], observations.aperture_area, observations.detector.exptime / observations.nsubexp, observations.masks.masks[:, :, n, :], A[:, :, tid], P[:, :, tid], p[:, :, tid], refraction, iffts[tid], convs[tid], object.background / observations.dim^2 / observations.nsubaps, atmosphere.transmission, observations.optics.response, ϕ_composite[:, :, tid], observations.phase_static, ϕ_slices[:, :, tid], atmosphere.phase, smoothing!, atmosphere.nlayers, extractors[(t-1)*observations.nsubexp + tsub, :, :, :], atmosphere.sampling_nyquist_mperpix, atmosphere.heights, patches.npatches, atmosphere.nλ, nλint, atmosphere.Δλ)
                next!(prog)
            end

            images_float[:, :, tid] .= max.(zero(FTYPE), images_float[:, :, tid])
            if noise == true
                add_noise!(images_float[:, :, tid], observations.detector.rn, true, FTYPE=FTYPE)
            end
            images_float[:, :, tid] .= min.(images_float[:, :, tid], observations.detector.saturation)
            images_float[:, :, tid] ./= observations.detector.gain  # Converts e⁻ to counts
            convert_image(observations.images[:, :, n, t], images_float[:, :, tid]) # Converts floating-point counts to integer at bitdepth of detector
        end
    end
    finish!(prog)
end

@views function create_radiant_energy_pre_detector!(image, image_small_temp, image_big_temp, psf::AbstractMatrix{<:AbstractFloat}, psf_temp, scale_psfs, object, patch_weight, object_patch, aperture_area, exptime, masks, A, P::AbstractMatrix{<:Complex{<:AbstractFloat}}, p::AbstractMatrix{<:Complex{<:AbstractFloat}}, refraction, iffts, convs, background, atmosphere_transmission, optics_response, ϕ_composite, phase_static, ϕ_slices, ϕ_full, smoothing!, nlayers, extractors, sampling_nyquist_mperpix, heights, npatches, nλ, nλint, Δλ)
    # fill!(image, zero(eltype(image)))
    for np=1:npatches
        for w₁=1:nλ
            for w₂=1:nλint
                w = (w₁-1)*nλint + w₂
                create_spectral_irradiance_at_aperture!(image_small_temp, image_big_temp, psf, psf_temp, scale_psfs[w], object[:, :, w], patch_weight[:, :, np], object_patch, masks[:, :, w], A, P, p, refraction[w], iffts, convs, background / (Δλ * nλ * nλint * npatches), atmosphere_transmission[w], optics_response[w], ϕ_composite, phase_static[:, :, w], ϕ_slices, ϕ_full[:, :, :, w], smoothing!, nlayers, extractors[np, :, w], sampling_nyquist_mperpix, heights, nλint)
                image .+= image_small_temp .* (Δλ * aperture_area * exptime)
            end
        end
    end
end

@views function create_radiant_energy_pre_detector!(image, image_small_temp, image_big_temp, psfs::AbstractArray{<:AbstractFloat, 4}, psf_temp, scale_psfs, object, patch_weight, object_patch, aperture_area, exptime, masks, A, P::AbstractArray{<:Complex{<:AbstractFloat}, 4}, p::AbstractArray{<:Complex{<:AbstractFloat}, 4}, refraction, iffts, conv!, background, atmosphere_transmission, optics_response, ϕ_composite, phase_static, ϕ_slices, ϕ_full, smoothing!, nlayers, extractors, sampling_nyquist_mperpix, heights, npatches, nλ, nλint, Δλ)
    for np=1:npatches
        for w₁=1:nλ
            for w₂=1:nλint 
                w = (w₁-1)*nλint + w₂
                create_spectral_irradiance_at_aperture!(image_small_temp, image_big_temp, psfs[:, :, np, w], psf_temp, scale_psfs[w], object[:, :, w], patch_weight[:, :, np], object_patch, masks[:, :, w], A, P[:, :, np, w], p[:, :, np, w], refraction[w], iffts, conv!, background  / (Δλ * nλ * nλint * npatches), atmosphere_transmission[w], optics_response[w], ϕ_composite, phase_static[:, :, w], ϕ_slices, ϕ_full[:, :, :, w], smoothing!, nlayers, extractors[np, :, w], sampling_nyquist_mperpix, heights, nλint)
                image .+= image_small_temp .* (Δλ * aperture_area * exptime)
            end
        end
    end
end

@views function create_spectral_irradiance_at_aperture!(image_small_temp, image_big_temp, psf, psf_temp, scale_psfs, object, patch_weight, object_patch, masks, A, P, p, refraction, iffts, conv!, background, atmosphere_transmission, optics_response, ϕ_composite, phase_static, ϕ_slices, ϕ_full, smoothing!, nlayers, extractors, sampling_nyquist_mperpix, heights, nλint)
    calculate_composite_pupil!(A, ϕ_composite, ϕ_slices, ϕ_full, nlayers, extractors, masks, sampling_nyquist_mperpix, heights)
    smoothing!(ϕ_composite, ϕ_composite)
    ϕ_composite .+= phase_static

    pupil2psf!(psf, psf_temp, masks, P, p, A, ϕ_composite, scale_psfs, iffts, refraction)
    psf ./= nλint

    object_patch .= patch_weight .* object
    conv!(image_big_temp, object_patch, psf)
    block_reduce!(image_small_temp, image_big_temp)
    
    image_small_temp .+= background
    image_small_temp .*= optics_response * atmosphere_transmission
end
