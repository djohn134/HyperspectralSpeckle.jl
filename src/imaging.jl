using Distributions
using TwoDimensional
using LinearInterpolators


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

function position2phase(ϕ_full, position, scaleby_height, scaleby_wavelength, dim; FTYPE=Float64)
    ϕ_out = zeros(FTYPE, dim, dim)
    extractor = create_extractor_operator(position, size(ϕ_full, 1), dim, scaleby_height, scaleby_wavelength, FTYPE=FTYPE)
    position2phase!(ϕ_out, ϕ_full, extractor)
    return ϕ_out
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

function pupil2psf(mask, λ, λ_ref, ζ, A, ϕ, build_dim, response, transmission, scale_psf, pixscale; FTYPE=Float64)
    P = zeros(FTYPE, build_dim, build_dim)
    p = zeros(Complex{FTYPE}, build_dim, build_dim)
    psf = zeros(FTYPE, build_dim, build_dim)
    psf_temp = zeros(FTYPE, build_dim, build_dim)
    refraction = create_refraction_operator(λ, λ_ref, ζ, pixscale, build_dim; FTYPE=FTYPE)
    pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, response, transmission, scale_psf, FTYPE(build_dim), refraction)
    return psf
end

function pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, response, transmission, scale_psf, scale_ifft::AbstractFloat, refraction)
    P .= mask .* scale_psf .* A .* cis.(ϕ)
    p .= ift(P) .* scale_ifft
    psf_temp .= transmission .* response .* abs2.(p)
    mul!(psf, refraction, psf_temp)
end

function pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, response, transmission, scale_psf, ifft_prealloc!::Function, refraction)
    P .= mask .* scale_psf .* A .* cis.(ϕ)
    ifft_prealloc!(p, P)
    psf_temp .= transmission .* response .* abs2.(p)
    mul!(psf, refraction, psf_temp)
end

@views function poly2broadbandpsfs(patches, observations, Δλ, nλ)
    FTYPE = gettype(observations[1])
    ndatasets = length(observations)
    patches.broadband_psfs = Vector{Array{FTYPE, 5}}(undef, ndatasets)
    for dd=1:ndatasets
        observation = observations[dd]
        psfs = patches.psfs[dd]
        patches.broadband_psfs[dd] = zeros(FTYPE, observation.dim, observation.dim, patches.npatches, observation.nsubaps, observation.nepochs)
        poly2broadbandpsfs!(patches.broadband_psfs[dd], psfs, patches, observation, Δλ, nλ)
    end
end

@views function poly2broadbandpsfs!(broadband_psfs, psfs, patches, observations, Δλ, nλ)
    Threads.@threads for t=1:observations.nepochs
        for n=1:observations.nsubaps
            for np=1:patches.npatches
                for w=1:nλ
                    broadband_psfs[:, :, np, n, t] .+= psfs[:, :, np, n, t, w] ./ nλ
                end
            end
        end
    end
end

function create_monochromatic_image(object, psf, dim)
    FTYPE = gettype(object)
    image_big = zeros(FTYPE, size(object))
    image_small = zeros(FTYPE, dim, dim)
    create_monochromatic_image!(image_small, image_big, object, psf)
    return image_small
end

function create_monochromatic_image!(image_small, image_big, object::AbstractMatrix{<:AbstractFloat}, psf)
    image_big .= conv_psf(object, psf)
    block_reduce!(image_small, image_big)
end

function create_monochromatic_image!(image_small, image_big, object::AbstractMatrix{<:AbstractFloat}, psf, conv_prealloc)
    conv_prealloc(image_big, object, psf)
    block_reduce!(image_small, image_big)
end

function create_monochromatic_image!(image_small, image_big, o_conv::Function, psf)
    image_big .= o_conv(psf)
    block_reduce!(image_small, image_big)
end

# function create_polychromatic_image(object, psfs, λ, Δλ, dim; FTYPE=Float64)
#     build_dim = size(psfs, 1)
#     image = zeros(FTYPE, build_dim, build_dim)
#     image_small = zeros(FTYPE, dim, dim)
#     image_big = zeros(FTYPE, build_dim, build_dim)
#     create_polychromatic_image!(image, image_small, image_big, object, psfs, λ, Δλ)
#     return image
# end

# @views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 2}, image_big, o_conv::AbstractVector{<:Function}, psfs, λ, Δλ)
#     nλ = length(λ)
#     for w=1:nλ
#         create_monochromatic_image!(image_small, image_big, o_conv[w], psfs[:, :, w])
#         image .+= image_small
#     end
#     image .*= Δλ
# end

# @views function create_polychromatic_image!(image, image_small::AbstractMatrix{<:AbstractFloat}, image_big, ω, object_patch, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ)
#     nλ = length(λ)
#     for w=1:nλ
#         object_patch .= ω .* object[:, :, w]
#         create_monochromatic_image!(image_small, image_big, object_patch, psfs[:, :, w])
#         image .+= image_small
#     end
#     image .*= Δλ
# end

# @views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 3}, image_big, ω, object_patch, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ)
#     nλ = length(λ)
#     for w=1:nλ
#         object_patch .= ω .* object[:, :, w]
#         create_monochromatic_image!(image_small[:, :, w], image_big, object_patch, psfs[:, :, w])
#         image .+= image_small[:, :, w]
#     end
#     image .*= Δλ
# end

function add_noise!(image, rn, poisson::Bool; FTYPE=Float64)
    if poisson == true
        image .= FTYPE.(rand.(Distributions.Poisson.(image)))
    end
    ## Read noise has a non-zero mean and sigma!
    image .+= rn .* randn(FTYPE, size(image))
    image .= max.(image, Ref(zero(FTYPE)))
end

function add_background!(image, background_flux; FTYPE=Float64)
    dim = size(image, 1)
    image .+= background_flux / dim^2
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

@views function create_images(patches, observations, atmosphere, masks, object; build_dim=object.dim, noise=false, verb=true)
    ndatasets = length(observations)
    FTYPE = gettype(patches)
    nthreads = Threads.nthreads()
    ndatasets = length(observations)
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

    scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    for dd=1:ndatasets
        if verb == true
            println("Creating $(observations[dd].dim)×$(observations[dd].dim) images for $(observations[dd].nepochs) times and $(observations[dd].nsubaps) subaps using eff model")
        end
        image_small_temp = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
        refraction = [create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale, build_dim, FTYPE=FTYPE) for w=1:atmosphere.nλ]
        extractors = create_patch_extractors(patches, atmosphere, observations[dd], object, scaleby_wavelength, scaleby_height, build_dim=build_dim)
        observations[dd].images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
        create_images!(patches, observations[dd], atmosphere, masks[dd], object, image_small_temp, image_big_temp, psfs, psf_temp, object_patch, A, P, p, refraction, iffts, convs, ϕ_composite, ϕ_slices, extractors, noise=noise)
    end
end

@views function create_images!(patches, observations, atmosphere, masks, object, image_small_temp, image_big_temp, psf, psf_temp, object_patch, A, P, p, refraction, iffts, convs, ϕ_composite, ϕ_slices, extractors; noise=false)
    FTYPE = gettype(observations)
    nλint = 1
    smoothing!(out, in) = nothing
    Threads.@threads :static for t=1:observations.nepochs
        tid = Threads.threadid()
        for n=1:observations.nsubaps
            create_image!(observations.images[:, :, n, t], image_small_temp[:, :, tid], image_big_temp[:, :, tid], psf[:, :, tid], psf_temp[:, :, tid], masks.scale_psfs, object.object, patches.w, object_patch[:, :, tid], masks.masks[:, :, n, :], A[:, :, tid], P[:, :, tid], p[:, :, tid], refraction, iffts[tid], convs[tid], atmosphere.transmission, observations.optics.response, ϕ_composite[:, :, tid], observations.phase_static, ϕ_slices[:, :, tid], atmosphere.phase, smoothing!, atmosphere.nlayers, extractors[t, :, :, :], atmosphere.sampling_nyquist_mperpix, atmosphere.heights, patches.npatches, atmosphere.nλ, nλint, atmosphere.Δλ)
            observations.images[:, :, n, t] .= max.(zero(FTYPE), observations.images[:, :, n, t])
            add_background!(observations.images[:, :, n, t], object.background_flux, FTYPE=FTYPE)
            if noise == true
                add_noise!(observations.images[:, :, n, t], observations.detector.rn, true, FTYPE=FTYPE)
            end
            observations.images[:, :, n, t] .= min.(observations.images[:, :, n, t], observations.detector.saturation)
        end
    end
end

@views function create_image!(image, image_small_temp, image_big_temp, psf::AbstractMatrix{<:AbstractFloat}, psf_temp, scale_psfs, object, patch_weight, object_patch, masks, A, P::AbstractMatrix{<:Complex{<:AbstractFloat}}, p::AbstractMatrix{<:Complex{<:AbstractFloat}}, refraction, iffts, convs, atmosphere_transmission, optics_response, ϕ_composite, phase_static, ϕ_slices, ϕ_full, smoothing!, nlayers, extractors, sampling_nyquist_mperpix, heights, npatches, nλ, nλint, Δλ)
    for np=1:npatches
        psf .*= 0
        for w₁=1:nλ
            for w₂=1:nλint 
                w = (w₁-1)*nλint + w₂
                create_patch_spectral_image!(image, image_small_temp, image_big_temp, psf, psf_temp, scale_psfs[w], object[:, :, w], patch_weight[:, :, np], object_patch, masks[:, :, w], A, P, p, refraction[w], iffts, convs, atmosphere_transmission[w], optics_response[w], ϕ_composite, phase_static[:, :, w], ϕ_slices, ϕ_full[:, :, :, w], smoothing!, nlayers, extractors[np, :, w], sampling_nyquist_mperpix, heights, nλint)
            end
        end
    end
    image .*= Δλ
end

@views function create_image!(image, image_small_temp, image_big_temp, psfs::AbstractArray{<:AbstractFloat, 4}, psf_temp, scale_psfs, object, patch_weight, object_patch, masks, A, P::AbstractArray{<:Complex{<:AbstractFloat}, 4}, p::AbstractArray{<:Complex{<:AbstractFloat}, 4}, refraction, iffts, convs, atmosphere_transmission, optics_response, ϕ_composite, phase_static, ϕ_slices, ϕ_full, smoothing!, nlayers, extractors, sampling_nyquist_mperpix, heights, npatches, nλ, nλint, Δλ)
    for np=1:npatches
        for w₁=1:nλ
            for w₂=1:nλint 
                w = (w₁-1)*nλint + w₂
                create_patch_spectral_image!(image, image_small_temp, image_big_temp, psfs[:, :, np, w], psf_temp, scale_psfs[w], object[:, :, w], patch_weight[:, :, np], object_patch, masks[:, :, w], A, P[:, :, np, w], p[:, :, np, w], refraction[w], iffts, convs, atmosphere_transmission[w], optics_response[w], ϕ_composite, phase_static[:, :, w], ϕ_slices, ϕ_full[:, :, :, w], smoothing!, nlayers, extractors[np, :, w], sampling_nyquist_mperpix, heights, nλint)
            end
        end
    end
    image .*= Δλ
end

@views function create_patch_spectral_image!(image, image_small_temp, image_big_temp, psf, psf_temp, scale_psfs, object, patch_weight, object_patch, masks, A, P, p, refraction, iffts, convs, atmosphere_transmission, optics_response, ϕ_composite, phase_static, ϕ_slices, ϕ_full, smoothing!, nlayers, extractors, sampling_nyquist_mperpix, heights, nλint)
    calculate_composite_pupil!(A, ϕ_composite, ϕ_slices, ϕ_full, nlayers, extractors, masks, sampling_nyquist_mperpix, heights)
    ϕ_composite .+= phase_static
    # ϕ_composite .*= masks
    smoothing!(ϕ_composite, ϕ_composite)
    pupil2psf!(psf_temp, psf_temp, masks, P, p, A, ϕ_composite, optics_response, atmosphere_transmission, scale_psfs, iffts, refraction)
    psf .= psf_temp ./ nλint
    object_patch .= patch_weight .* object
    create_monochromatic_image!(image_small_temp, image_big_temp, object_patch, psf, convs)
    image .+= image_small_temp
end
