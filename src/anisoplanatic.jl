abstract type AbstractPatches end
function display(patches::T) where {T<:AbstractPatches}
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Anisoplanatic Patches\n"); print(Crayon(reset=true))
    println("\tSize: $(patches.dim) pixels")
    println("\tOverlap: $(patches.overlap)")
    println("\tNumber of patches: $(Int(sqrt(patches.npatches)))×$(Int(sqrt(patches.npatches))) patches")
end

mutable struct AnisoplanaticPatches{T<:AbstractFloat}  <: AbstractPatches # Structure to store isoplanatic patch info
    npatches::Int64  # Number of patches per image side length, e.g. 7x7 patches per images -> npatches_per_image_side = 7
    dim::Int64      # Patch width, e.g. 64x64 pixel patch -> npix_isopatch_width = 64
    overlap::T         # Percentage of window patch overlap, e.g. 50% overlap -> patch_overlap = 0.5
    coords::Matrix{Vector{Int64}}  # Patch lower and upper bounds
    positions::Matrix{T} # Locations of the patch centers, relative to the image center
    w::Array{T, 3}                  # Windowing function
    @views function AnisoplanaticPatches(
            patch_dim, 
            image_dim;
            isoplanatic=false,
            verb=true, 
            FTYPE=Float64, 
            ITYPE=Int64
        )
        patch_dim = (isoplanatic == true) ? image_dim : patch_dim
        overlap = (isoplanatic == true) ? 0.0 : 0.5
        npatches_side = (isoplanatic == true) ? 1 : ceil(ITYPE, image_dim / (patch_dim * (1 - overlap))) + 1        
        npatches = npatches_side^2
        xcenters = round.(ITYPE, [-npatches_side÷2:npatches_side÷2;] .* patch_dim .* (1-overlap))
        xstarts = xcenters .- (patch_dim÷2)
        xends = xcenters .+ (patch_dim÷2-1)
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
            mask[yrange, xrange] .= 1
            w[:, :, n] .= (isoplanatic == true) ? 1.0 : mask .* bartlett_hann2d(xx .- (image_dim - patch_dim)/2 .- positions[1, n], yy.- (image_dim - patch_dim)/2 .- positions[2, n], patch_dim)
        end
        patches = new{FTYPE}(npatches, patch_dim, overlap, coords, positions, w)
        if verb == true
            display(patches)
        end
        return patches
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
