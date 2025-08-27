using OhMyThreads
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

<<<<<<< HEAD
function pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, scale_psf, ifft_prealloc!::Function, refraction)
    P .= mask .* scale_psf .* A .* cis.(ϕ)
    ifft_prealloc!(p, P)
    psf_temp .= abs2.(p)
    fftshift!(psf, psf_temp)
    mul!(psf_temp, refraction, psf)
    fftshift!(psf, psf_temp)
end

=======
function pupil2psf(mask, λ, λ_ref, ζ, A, ϕ, build_dim, α, scale_psf, pixscale; FTYPE=Float64)
    P = zeros(FTYPE, build_dim, build_dim)
    p = zeros(Complex{FTYPE}, build_dim, build_dim)
    psf = zeros(FTYPE, build_dim, build_dim)
    psf_temp = zeros(FTYPE, build_dim, build_dim)
    refraction = create_refraction_operator(λ, λ_ref, ζ, pixscale, build_dim; FTYPE=FTYPE)
    pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, α, scale_psf, FTYPE(build_dim), refraction)
    return psf
end

function pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, α, scale_psf, scale_ifft::AbstractFloat, refraction)
    P .= mask .* scale_psf .* A .* cis.(ϕ)
    p .= ift(P) .* scale_ifft
    psf_temp .= α .* abs2.(p)
    mul!(psf, refraction, psf_temp)
end

function pupil2psf!(psf, psf_temp, mask, P, p, A, ϕ, α, scale_psf, ifft_prealloc!::Function, refraction)
    P .= mask .* scale_psf .* A .* cis.(ϕ)
    ifft_prealloc!(p, P)
    psf_temp .= α .* abs2.(p)
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
    image_big .= conv_prealloc(object, psf)
    block_reduce!(image_small, image_big)
end

function create_monochromatic_image!(image_small, image_big, o_conv::Function, psf)
    image_big .= o_conv(psf)
    block_reduce!(image_small, image_big)
end

function create_polychromatic_image(object, psfs, λ, Δλ, dim; FTYPE=Float64)
    build_dim = size(psfs, 1)
    image = zeros(FTYPE, build_dim, build_dim)
    image_small = zeros(FTYPE, dim, dim)
    image_big = zeros(FTYPE, build_dim, build_dim)
    create_polychromatic_image!(image, image_small, image_big, object, psfs, λ, Δλ)
    return image
end

# @views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 2}, image_big, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ)
#     println("B")
#     nλ = length(λ)
#     for w=1:nλ
#         create_monochromatic_image!(image_small, image_big, object[:, :, w], psfs[:, :, w])
#         image .+= image_small
#     end
#     image .*= Δλ
# end

# @views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 2}, image_big, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ, conv_prealloc)
#     println("C")
#     nλ = length(λ)
#     for w=1:nλ
#         create_monochromatic_image!(image_small, image_big, object[:, :, w], psfs[:, :, w], conv_prealloc)
#         image .+= image_small
#     end
#     image .*= Δλ
# end

@views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 2}, image_big, o_conv::AbstractVector{<:Function}, psfs, λ, Δλ)
    # OPD solve step
    nλ = length(λ)
    for w=1:nλ
        create_monochromatic_image!(image_small, image_big, o_conv[w], psfs[:, :, w])
        image .+= image_small
    end
    image .*= Δλ
end

# @views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 3}, image_big, object, psfs, λ, Δλ)
#     println("E")
#     nλ = length(λ)
#     for w=1:nλ
#         create_monochromatic_image!(image_small[:, :, w], image_big, object[:, :, w], psfs[:, :, w])
#         image .+= image_small[:, :, w]
#     end
#     image .*= Δλ
# end

@views function create_polychromatic_image!(image, image_small::AbstractMatrix{<:AbstractFloat}, image_big, ω, object_patch, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ)
    # Object solve step
    nλ = length(λ)
    for w=1:nλ
        object_patch .= ω .* object[:, :, w]
        create_monochromatic_image!(image_small, image_big, object_patch, psfs[:, :, w])
        image .+= image_small
    end
    image .*= Δλ
end

# @views function create_polychromatic_image!(image, image_small::AbstractMatrix{<:AbstractFloat}, image_big, ω, object_patch, object::AbstractMatrix{<:AbstractFloat}, psfs, λ, Δλ)
#     println("G")
#     nλ = length(λ)
#     for w=1:nλ
#         object_patch .= ω .* object
#         create_monochromatic_image!(image_small, image_big, object_patch, psfs[:, :, w])
#         image .+= image_small
#     end
#     image .*= Δλ
# end

@views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 3}, image_big, ω, object_patch, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ)
    # Data generation step
    nλ = length(λ)
    for w=1:nλ
        object_patch .= ω .* object[:, :, w]
        create_monochromatic_image!(image_small[:, :, w], image_big, object_patch, psfs[:, :, w])
        image .+= image_small[:, :, w]
    end
    image .*= Δλ
end

# @views function create_polychromatic_image!(image, image_small::AbstractMatrix{<:AbstractFloat}, image_big, ω, object_patch, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ, conv_prealloc)
#     println("I")
#     nλ = length(λ)
#     for w=1:nλ
#         object_patch .= ω .* object[:, :, w]
#         create_monochromatic_image!(image_small, image_big, object_patch, psfs[:, :, w], conv_prealloc)
#         image .+= image_small
#     end
#     image .*= Δλ
# end

# @views function create_polychromatic_image!(image, image_small::AbstractMatrix{<:AbstractFloat}, image_big, ω, object_patch, object::AbstractMatrix{<:AbstractFloat}, psfs, λ, Δλ, conv_prealloc)
#     println("J")
#     nλ = length(λ)
#     for w=1:nλ
#         object_patch .= ω .* object
#         create_monochromatic_image!(image_small, image_big, object_patch, psfs[:, :, w], conv_prealloc)
#         image .+= image_small
#     end
#     image .*= Δλ
# end

# @views function create_polychromatic_image!(image, image_small::AbstractArray{<:AbstractFloat, 3}, image_big, ω, object_patch, object::AbstractArray{<:AbstractFloat, 3}, psfs, λ, Δλ, conv_prealloc)
#     println("K")
#     nλ = length(λ)
#     for w=1:nλ
#         object_patch .= ω .* object[:, :, w]
#         create_monochromatic_image!(image_small[:, :, w], image_big, object_patch, psfs[:, :, w], conv_prealloc)
#         image .+= image_small[:, :, w]
#     end
#     image .*= Δλ
# end

>>>>>>> main
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
        mul!(ϕ_slices, extractors[l], ϕ_full[:, :, l])
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
    channel_builddim_real = Channel{Matrix{FTYPE}}(6 * nthreads)
    channel_builddim_real_4d = Channel{Array{FTYPE, 4}}(2 * nthreads)
    channel_builddim_cplx = Channel{Matrix{Complex{FTYPE}}}(2 * nthreads)
    channel_builddim_cplx_4d = Channel{Array{Complex{FTYPE}, 4}}(2 * nthreads)
    channel_builddim_ones = Channel{Matrix{Complex{FTYPE}}}(nthreads)
    channel_imagedim = Channel{Matrix{FTYPE}}(2 * nthreads)
    channel_iffts = Channel{Function}(nthreads)
    channel_convs = Channel{ConvolutionPlan{FTYPE}}(nthreads)
    channel_smooth = Channel{Nothing}(nthreads)
    foreach(1:6*nthreads) do ~
        put!(channel_builddim_real, zeros(FTYPE, build_dim, build_dim))
    end

    foreach(1:2*nthreads) do ~
        put!(channel_imagedim, zeros(FTYPE, observations.dim, observations.dim))
        put!(channel_builddim_cplx, zeros(Complex{FTYPE}, build_dim, build_dim))
        put!(channel_builddim_cplx_4d, zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, object.nλ))
    end

    foreach(1:nthreads) do ~
        put!(channel_builddim_real_4d, zeros(FTYPE, build_dim, build_dim, patches.npatches, object.nλ))
        put!(channel_iffts, setup_ifft(Complex{FTYPE}, build_dim)[1])
        put!(channel_convs, ConvolutionPlan(build_dim, FTYPE=FTYPE))
        put!(channel_builddim_ones, ones(FTYPE, build_dim, build_dim))
        put!(channel_smooth, nothing)
    end
    channels = (; channel_builddim_real, channel_builddim_real_4d, channel_builddim_cplx, channel_builddim_cplx_4d, channel_builddim_ones, channel_imagedim, channel_iffts, channel_convs, channel_smooth)
    
    scaleby_height = layer_scale_factors(atmosphere.heights, object.range)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    DTYPE = gettypes(observations.detector)[end]
    refraction = [create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations.ζ, observations.detector.pixscale, build_dim, FTYPE=FTYPE) for w=1:atmosphere.nλ]
    extractors = create_patch_extractors(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height, build_dim=build_dim)    
    observations.images = zeros(DTYPE, observations.dim, observations.dim, observations.nsubaps, observations.nepochs)
    if verb == true
        println("Creating $(observations.dim)×$(observations.dim) images for $(observations.nepochs) times ($(observations.nsubexp) subexposures each) and $(observations.nsubaps) subaps")
    end
    create_detector_images!(patches, observations, atmosphere, object, refraction, extractors, channels, noise=noise)
end

function take_buffers(channels)
    image_float = take!(channels.channel_imagedim)
    image_small = take!(channels.channel_imagedim)
    image_big = take!(channels.channel_builddim_real)
    psf = take!(channels.channel_builddim_real_4d)
    psf_temp = take!(channels.channel_builddim_real)
    object_patch = take!(channels.channel_builddim_real)
    A = take!(channels.channel_builddim_ones)
    P = take!(channels.channel_builddim_cplx_4d)
    p = take!(channels.channel_builddim_cplx_4d)
    ϕ_composite = take!(channels.channel_builddim_real)
    ϕ_slices = take!(channels.channel_builddim_real)
    iffts = take!(channels.channel_iffts)
    conv_plan = take!(channels.channel_convs)
    smooth = take!(channels.channel_smooth)
    zeros!(image_float)
    return (; image_float, image_small, image_big, psf, psf_temp, object_patch, A, P, p, ϕ_composite, ϕ_slices, iffts, conv_plan, smooth)
end

function put_buffers(buffers, channels)
    put!(channels.channel_imagedim, buffers.image_float)
    put!(channels.channel_imagedim, buffers.image_small)
    put!(channels.channel_builddim_real, buffers.image_big)
    put!(channels.channel_builddim_real_4d, buffers.psf)
    put!(channels.channel_builddim_real, buffers.psf_temp)
    put!(channels.channel_builddim_real, buffers.object_patch)
    put!(channels.channel_builddim_ones, buffers.A)
    put!(channels.channel_builddim_cplx_4d, buffers.P)
    put!(channels.channel_builddim_cplx_4d, buffers.p)
    put!(channels.channel_builddim_real, buffers.ϕ_composite)
    put!(channels.channel_builddim_real, buffers.ϕ_slices)
    put!(channels.channel_iffts, buffers.iffts)
    put!(channels.channel_convs, buffers.conv_plan)
    put!(channels.channel_smooth, buffers.smooth)
end

@views function create_detector_images!(patches, observations, atmosphere, object, refraction, extractors, channels; noise=false)
    FTYPE = gettype(observations)
    prog = Progress(observations.nepochs*observations.nsubexp*observations.nsubaps)
    tforeach(collect(Iterators.product(1:observations.nepochs, 1:observations.nsubaps))) do (t, n)
        buffers = take_buffers(channels)
        for tsub=1:observations.nsubexp 
            create_radiant_energy_pre_detector!(buffers.image_float, observations, object, atmosphere, patches, refraction, extractors[(t-1)*observations.nsubexp + tsub, :, :, :], buffers, (; n, t))
            next!(prog)
        end

        buffers.image_float .= max.(zero(FTYPE), buffers.image_float)
        if noise == true
            add_noise!(buffers.image_float, observations.detector.rn, true, FTYPE=FTYPE)
        end
        buffers.image_float .= min.(buffers.image_float, observations.detector.saturation)
        buffers.image_float ./= observations.detector.gain  # Converts e⁻ to counts
        convert_image(observations.images[:, :, n, t], buffers.image_float) # Converts floating-point counts to integer at bitdepth of detector
        put_buffers(buffers, channels)
    end
    finish!(prog)
end

@views function create_radiant_energy_pre_detector!(image, observations, object, atmosphere, patches, refraction, extractors, buffers, ixs)
    for np=1:patches.npatches
        for w₁=1:atmosphere.nλ
            for w₂=1:object.nλint
                w = (w₁-1)*object.nλint + w₂
                create_spectral_irradiance_at_aperture!(observations, object, atmosphere, patches, refraction, extractors, buffers, (; ixs..., np, w))
                image .+= buffers.image_small .* (atmosphere.Δλ * observations.aperture_area * observations.detector.exptime)
            end
        end
    end
end

@views function create_spectral_irradiance_at_aperture!(observations, object, atmosphere, patches, refraction, extractors, buffers, ixs)
    calculate_composite_pupil!(buffers.A, buffers.ϕ_composite, buffers.ϕ_slices, atmosphere.phase[:, :, :, ixs.w], atmosphere.nlayers, extractors[ixs.np, :, ixs.w], observations.masks.masks[:, :, ixs.n, ixs.w], atmosphere.sampling_nyquist_mperpix, atmosphere.heights)
    if !isnothing(buffers.smooth)
        convolve!(buffers.ϕ_composite, buffers.smooth, buffers.ϕ_composite)
    end
    buffers.ϕ_composite .+= observations.phase_static[:, :, ixs.w]
    
    pupil2psf!(buffers.psf[:, :, ixs.np, ixs.w], buffers.psf_temp, observations.masks.masks[:, :, ixs.n, ixs.w], buffers.P[:, :, ixs.np, ixs.w], buffers.p[:, :, ixs.np, ixs.w], buffers.A, buffers.ϕ_composite, observations.masks.scale_psfs[ixs.w], buffers.iffts, refraction[ixs.w])
    buffers.psf[:, :, ixs.np, ixs.w] ./= object.nλint

    buffers.object_patch .= patches.w[:, :, ixs.np] .* object.object[:, :, ixs.w]
    convolve!(buffers.image_big, buffers.conv_plan, buffers.object_patch, buffers.psf[:, :, ixs.np, ixs.w])
    block_reduce!(buffers.image_small, buffers.image_big)

    buffers.image_small .+= object.background / (atmosphere.Δλ * atmosphere.nλ * object.nλint * patches.npatches)
    buffers.image_small .*= observations.optics.response[ixs.w] * atmosphere.transmission[ixs.w]
end
