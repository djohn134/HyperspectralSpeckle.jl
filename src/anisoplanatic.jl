mutable struct AnisoplanaticPatches{T<:AbstractFloat}  # Structure to store isoplanatic patch info
    npatches::Int64  # Number of patches per image side length, e.g. 7x7 patches per images -> npatches_per_image_side = 7
    dim::Int64      # Patch width, e.g. 64x64 pixel patch -> npix_isopatch_width = 64
    overlap::T         # Percentage of window patch overlap, e.g. 50% overlap -> patch_overlap = 0.5
    coords::Matrix{Vector{Int64}}  # Patch lower and upper bounds
    positions::Matrix{T} # Locations of the patch centers, relative to the image center
    w::Array{T, 3}                  # Windowing function
    A::Vector{Array{T, 6}}
    phase_slices::Array{T, 6}
    phase_composite::Array{T, 5}
    psfs::Vector{Array{T, 6}}
    @views function AnisoplanaticPatches(
            dim, 
            image_dim, 
            overlap; 
            isoplanatic=false,
            verb=true, 
            FTYPE=Float64, 
            ITYPE=Int64
        )
        if isoplanatic == true
            dim = image_dim
            overlap = 0    
        end
        npatches_side = dim==image_dim ? 1 : ceil(ITYPE, image_dim / (dim * (1 - overlap))) + 1

        if verb == true
            print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Efficient filter flow\n"); print(Crayon(reset=true))
            println("\tSize: $(dim) pixels")
            println("\tOverlap: $(overlap)")
            println("\tNumber of patches: $(npatches_side)x$(npatches_side) patches")
        end
        
        npatches = npatches_side^2
        xcenters = round.(ITYPE, [-npatches_side÷2:npatches_side÷2;] .* dim .* (1-overlap))
        xstarts = xcenters .- (dim÷2)
        xends = xcenters .+ (dim÷2-1)
        positions = [[i for i in xcenters for j in xcenters] [j for i in xcenters for j in xcenters]]';
        xpos = [[xstarts[n] .+ (image_dim÷2+1), xends[n] .+ (image_dim÷2+1)] for n=1:npatches_side];
        coords = hcat([i for i in xpos for j in xpos], [j for i in xpos for j in xpos]);
        w = zeros(FTYPE, image_dim, image_dim, npatches);
        xx = repeat([0:image_dim-1;], 1, image_dim); 
        yy = xx';
        mask = zeros(FTYPE, image_dim, image_dim)
        for n=1:npatches
            fill!(mask, zero(FTYPE))
            xrange = max(coords[n, 2][1], 1):min(coords[n, 2][2], image_dim)
            yrange = max(coords[n, 1][1], 1):min(coords[n, 1][2], image_dim)
            if isoplanatic == true
                w[:, :, n] .= 1
            else
                mask[yrange, xrange] .= 1
                w[:, :, n] .= mask .* bartlett_hann2d(xx .- (image_dim - dim)/2 .- positions[1, n], yy.- (image_dim - dim)/2 .- positions[2, n], dim)
            end
        end

        return new{FTYPE}(npatches, dim, overlap, coords, positions, w)
    end
end

function get_center(field_point, pupil_position, object_sampling_arcsecperpix, atmosphere_sampling_nyquist_mperpix, height, scaleby_wavelength)
    center = similar(field_point)
    get_center!(center, field_point, pupil_position, object_sampling_arcsecperpix, atmosphere_sampling_nyquist_mperpix, height, scaleby_wavelength)
    return center
end

function get_center!(center, field_point, pupil_position, object_sampling_arcsecperpix, atmosphere_sampling_nyquist_mperpix, height, scaleby_wavelength)
    center .= field_point  # Δx [pix in source plane]
    center .*= object_sampling_arcsecperpix / 206265  # Δx [radians]
    center .*= height*1000  # Δx [meters in layer]
    center ./= atmosphere_sampling_nyquist_mperpix  # Δx  [pix in layer]
    center .*= scaleby_wavelength
    center .+= pupil_position
end

@views function create_patch_extractors(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height; build_dim=observations.dim)
    FTYPE = gettype(atmosphere)
    extractor = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, observations.nepochs, patches.npatches, atmosphere.nlayers, atmosphere.nλ)
    Threads.@threads for t=1:observations.nepochs
        for np=1:patches.npatches
            for w=1:atmosphere.nλ
                for l=1:atmosphere.nlayers
                    center = get_center(patches.positions[:, np], atmosphere.positions[:, t, l, w], object.sampling_arcsecperpix, atmosphere.sampling_nyquist_mperpix[l], atmosphere.heights[l], scaleby_wavelength[w])
                    extractor[t, np, l, w] = create_extractor_operator(center, atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                end
            end
        end
    end

    return extractor
end

@views function create_patch_extractors_adjoint(patches, atmosphere, observations, object, scaleby_wavelength, scaleby_height; build_dim=observations.dim)
    FTYPE = gettype(atmosphere)
    extractor_adj = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, observations.nepochs, patches.npatches, atmosphere.nlayers, atmosphere.nλ)
    Threads.@threads for t=1:observations.nepochs
        for np=1:patches.npatches
            for w=1:atmosphere.nλ
                for l=1:atmosphere.nlayers
                    center = get_center(patches.positions[:, np], atmosphere.positions[:, t, l, w], object.sampling_arcsecperpix, atmosphere.sampling_nyquist_mperpix[l], atmosphere.heights[l], scaleby_wavelength[w])
                    extractor_adj[t, np, l, w] = create_extractor_adjoint(center, atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                end
            end
        end
    end

    return extractor_adj
end

@views function calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full; verb=true)
    if verb == true
        println("Creating sausage masks for $(atmosphere.nlayers) layers at $(atmosphere.nλ) wavelengths")
    end
    FTYPE = gettype(atmosphere)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
    buffer = zeros(FTYPE, atmosphere.dim, atmosphere.dim, Threads.nthreads())
    layer_mask = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, Threads.nthreads())
    deextractors = create_patch_extractors_adjoint(patches, atmosphere, observations_full, object, scaleby_wavelength, scaleby_height)
    Threads.@threads :static for t=1:observations_full.nepochs
        tid = Threads.threadid()
        for np=1:patches.npatches
            for n=1:observations_full.nsubaps
                for w=1:atmosphere.nλ
                    for l=1:atmosphere.nlayers
                        position2phase!(buffer[:, :, tid], masks_full.masks[:, :, n, w], deextractors[t, np, l, w])
                        layer_mask[:, :, l, w, tid] .+= buffer[:, :, tid]
                    end
                end
            end
        end
    end
    atmosphere.masks = dropdims(sum(layer_mask, dims=5), dims=5)
    atmosphere.masks[atmosphere.masks .> 0] .= 1
end

@views function calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks; build_dim=observations.dim, verb=true)
    if verb == true
        println("Calculating composite complex pupil for $(length(observations)) channels")
    end
    FTYPE = gettype(atmosphere)
    ndatasets = length(observations)
    patches.A = Vector{Array{FTYPE, 6}}(undef, ndatasets)
    patches.phase_slices = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[end].nepochs, atmosphere.nlayers, atmosphere.nλ)
    patches.phase_composite = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[end].nepochs, atmosphere.nλ)
    for dd=1:ndatasets
        patches.A[dd] = ones(FTYPE, build_dim, build_dim, observations[dd].nsubaps, patches.npatches, observations[dd].nepochs, atmosphere.nλ)
    end

    scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
    scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
    extractors = create_patch_extractors(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height, build_dim=build_dim)
    if verb == true
        println("\tExtracting composite phase")
    end
    calculate_composite_phase_eff!(patches.phase_composite, patches.phase_slices, patches, atmosphere, observations[end], masks[end], extractors)
    if atmosphere.propagate == true
        for dd=1:ndatasets
            if verb == true
                println("\tExtracting composite amplitude for channel $(dd) by Fresnel propagation")
            end
            calculate_composite_amplitude_eff!(patches.A[dd], masks[dd], observations[dd], atmosphere, patches)
        end
    end
end

@views function calculate_composite_phase_eff!(ϕ_composite, ϕ_slices, patches, atmosphere, observations, masks, extractors)
    FTYPE = gettype(atmosphere)
    fill!(ϕ_composite, zero(FTYPE))
    Threads.@threads for t=1:observations.nepochs
        for np=1:patches.npatches
            for w=1:atmosphere.nλ
                for l=1:atmosphere.nlayers
                    if atmosphere.common_opd == true
                        position2phase!(ϕ_slices[:, :, np, t, l, w], FTYPE(2pi) ./ atmosphere.λ[w] .* atmosphere.opd[:, :, l], extractors[t, np, l, w])
                    else
                        position2phase!(ϕ_slices[:, :, np, t, l, w], atmosphere.phase[:, :, l, w], extractors[t, np, l, w])
                    end
                    # ϕ_slices .*= masks.masks[:, :, 1, w]
                    ϕ_composite[:, :, np, t, w] .+= ϕ_slices[:, :, np, t, l, w]
                end
            end
        end
    end
end

@views function calculate_composite_amplitude_eff!(A, masks, observations, atmosphere, patches)
    FTYPE = gettype(atmosphere)
    Threads.@threads for t=1:observations.nepochs
        for np=1:patches.npatches
            for w=1:atmosphere.nλ
                for n=1:observations.nsubaps
                    ϕ_slices_subap = repeat(masks.masks[:, :, n, w], 1, 1, atmosphere.nlayers) .* patches.phase_slices[:, :, np, t, :, w]
                    N = size(patches.phase_slices, 1)
                    x1 = ((-N÷2:N÷2-1) .* atmosphere.sampling_nyquist_mperpix[1])' .* ones(N)
                    y1 = x1'
                    sg = repeat(exp.(-(x1 ./ (0.47*N)).^16) .* exp.(-(y1 ./ (0.47*N)).^16), 1, 1, atmosphere.nlayers)
                    Uout = propagate_layers(ones(Complex{FTYPE}, size(masks.masks[:, :, n, w])), atmosphere.λ[w], atmosphere.sampling_nyquist_mperpix[1], atmosphere.sampling_nyquist_mperpix[end], atmosphere.heights, sg .* cis.(ϕ_slices_subap))
                    A[:, :, n, np, t, w] .= masks.masks[:, :, n, w] .* abs.(Uout)
                end
            end
        end
    end
end

@views function create_images_eff(patches, observations, atmosphere, masks, object; build_dim=object.dim, noise=false, verb=true)
    ndatasets = length(observations)
    if verb == true
        for dd=1:ndatasets
            println("Creating $(observations[dd].dim)×$(observations[dd].dim) images for $(observations[dd].nepochs) times and $(observations[dd].nsubaps) subaps using eff model")
        end
    end
    FTYPE = gettype(patches)
    nthreads = Threads.nthreads()
    ndatasets = length(observations)
    psf_temp = zeros(FTYPE, build_dim, build_dim, nthreads)
    image_temp_big = zeros(FTYPE, build_dim, build_dim, nthreads)
    object_patch = zeros(FTYPE, build_dim, build_dim, nthreads)
    P = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    p = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
    iffts = [setup_ifft(build_dim, FTYPE=FTYPE) for tid=1:Threads.nthreads()]
    
    refraction = Matrix{Any}(undef, ndatasets, atmosphere.nλ)
    patches.psfs = Vector{Array{FTYPE, 6}}(undef, ndatasets)
    for dd=1:ndatasets
        refraction[dd, :] .= [create_refraction_operator(atmosphere.λ[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale, build_dim, FTYPE=FTYPE) for w=1:atmosphere.nλ]
        patches.psfs[dd] = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, atmosphere.nλ)
        observations[dd].images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
        observations[dd].monochromatic_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs, atmosphere.nλ)
    end

    create_images_eff!(patches, observations, atmosphere, masks, object, object_patch, psf_temp, image_temp_big, P, p, iffts, refraction, noise=noise)
end

@views function create_images_eff!(patches, observations, atmosphere, masks, object, object_patch, psf_temp, image_temp_big, P, p, iffts, refraction; noise=false)
    FTYPE = gettype(patches)
    ndatasets = length(observations)
    for dd=1:ndatasets
        mask = masks[dd]
        observation = observations[dd]
        psfs = patches.psfs[dd]
        A = patches.A[dd]
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            for n=1:observation.nsubaps
                for np=1:patches.npatches
                    if sum(object.object .* patches.w[:, :, np]) > 0
                        for w=1:atmosphere.nλ
                            pupil2psf!(psfs[:, :, np, n, t, w], psf_temp[:, :, tid], mask.masks[:, :, n, w], P[:, :, tid], p[:, :, tid], A[:, :, n, np, t, w], patches.phase_composite[:, :, np, t, w] .+ observation.phase_static[:, :, w], observation.optics.response[w], atmosphere.transmission[w], mask.scale_psfs[w], iffts[tid], refraction[dd, w])
                        end
                        create_polychromatic_image!(observation.images[:, :, n, t], observation.monochromatic_images[:, :, n, t, :], image_temp_big[:, :, tid], patches.w[:, :, np], object_patch[:, :, tid], object.object, psfs[:, :, np, n, t, :], atmosphere.λ, atmosphere.Δλ)
                        observation.images[:, :, n, t] .= max.(0.0, observation.images[:, :, n, t])
                    end
                end
                add_background!(observation.images[:, :, n, t], object.background_flux, FTYPE=FTYPE)
                if noise == true
                    add_noise!(observation.images[:, :, n, t], observation.detector.rn, true, FTYPE=FTYPE)
                end
                observation.images[:, :, n, t] .= min.(observation.images[:, :, n, t], observation.detector.saturation)
            end
        end
    end
end

@views function change_heights!(patches, atmosphere, object, observations_full, masks_full, heights; reconstruction=[], verb=true)
    if verb == true
        println("Heights changed from $(atmosphere.heights) km to $(heights) km")
    end
    
    FTYPE = gettype(observations_full)
    original_heights = atmosphere.heights
    original_dim = atmosphere.dim
    original_sampling_nyquist_arcsecperpix = atmosphere.sampling_nyquist_arcsecperpix
    atmosphere.heights = heights
    atmosphere.sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(observations_full.D, object.fov, heights, observations_full.dim)

    calculate_screen_size!(atmosphere, observations_full, object, patches, verb=verb)
    calculate_pupil_positions!(atmosphere, observations_full, verb=verb)
    calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full, verb=verb)
    opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
    # ϕ = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)

    scaleby_height = original_sampling_nyquist_arcsecperpix ./ atmosphere.sampling_nyquist_arcsecperpix
    for l=1:atmosphere.nlayers
        kernel = LinearSpline(FTYPE)
        transform = AffineTransform2D{FTYPE}()
        screen_size = (Int64(original_dim), Int64(original_dim))
        output_size = (Int64(atmosphere.dim), Int64(atmosphere.dim))
        full_transform = ((transform + (screen_size[1]÷2+1, screen_size[2]÷2+1)) * (1/scaleby_height[l])) - (atmosphere.dim÷2+1, atmosphere.dim÷2+1)
        scaler = TwoDimensionalTransformInterpolator(output_size, screen_size, kernel, full_transform)
        mul!(opd[:, :, l], scaler, atmosphere.opd[:, :, l])            
    end
    atmosphere.opd = opd
    atmosphere.opd .*= atmosphere.masks[:, :, :, 1]

    if reconstruction != []
        helpers = reconstruction.helpers
        patch_helpers = reconstruction.patch_helpers
        if helpers != []
            helpers = reconstruction.helpers
            helpers.g_threads_opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, Threads.nthreads())
            helpers.phase_full = zeros(FTYPE, atmosphere.dim, atmosphere.dim, Threads.nthreads())
            helpers.containers_sdim_real = zeros(FTYPE, atmosphere.dim, atmosphere.dim, Threads.nthreads())
        end

        if patch_helpers != []
            scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
            scaleby_wavelength = atmosphere.λ_nyquist ./ atmosphere.λ
            patch_helpers.extractor = create_patch_extractors(patches, atmosphere, observations_full, object, scaleby_wavelength, scaleby_height)
            patch_helpers.extractor_adj = create_patch_extractors_adjoint(patches, atmosphere, observations_full, object, scaleby_wavelength, scaleby_height)
        end
    end
end
