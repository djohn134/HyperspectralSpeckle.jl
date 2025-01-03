using FFTW
using Statistics
using FourierTools
using LinearAlgebra
using OptimPackNextGen


function ConstantSchedule(fwhm)
    function constant(x)
        return fwhm
    end

    return constant
end

function LinearSchedule(maxval, niters, minval)
    m = (minval - maxval) / (niters - 1)
    b = maxval - m
    function linear(x)
        return max(m*x + b, minval)
    end

    return linear
end

function ReciprocalSchedule(maxval, minval)
    function reciprocal(x)
        return max(maxval * (1 / x), minval)
    end

    return reciprocal
end

mutable struct Helpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    refraction::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    refraction_adj::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    ft::Vector{Vector{Function}}
    ift::Vector{Vector{Function}}
    convolve::Vector{Vector{Function}}
    correlate::Vector{Vector{Function}}
    autocorr::Vector{Vector{Function}}
    ϕ_full::Array{T, 3}
    ϕ_slice::Array{T, 3}
    ϕ_composite::Array{T, 3}
    o_conv::Array{Function, 3}
    o_corr::Array{Function, 3}
    smoothing_kernel::Matrix{T}
    k_conv::Vector{Function}
    k_corr::Vector{Function}
    P::Array{Complex{T}, 4}
    p::Array{Complex{T}, 4}
    c::Array{T, 3}
    d::Array{Complex{T}, 3}
    d2::Array{T, 3}
    Î_big::Array{T, 3}
    Î_small::Vector{Array{T, 3}}
    r::Vector{Array{T, 3}}
    ω::Vector{Array{T, 3}}
    mask_acf::Vector{Matrix{T}}
    ϵ_threads::Vector{T}
    g_threads_obj::Array{T, 4}
    g_threads_opd::Array{T, 4}
    g_threads_ϕ::Array{T, 5}
    containers_builddim_real::Array{T, 3}
    containers_builddim_cplx::Array{Complex{T}, 3}
    containers_sdim_real::Array{T, 3}
    containers_sdim_cplx::Array{T, 3}
    containers_pdim_real::Vector{Array{T, 3}}
    containers_pdim_cplx::Vector{Array{Complex{T}, 3}}
    function Helpers(atmosphere,
                     observations,
                     object,
                     patches;
                     λtotal=atmosphere.λ,
                     build_dim=size(object.object, 1),
                     ndatasets=length(observations),
                     verb=true,
                     FTYPE=gettype(atmosphere))
        if verb == true
            println("Creating fft/ifft plans and convolution/correlation plans")
        end
        nλtotal = length(λtotal)
        fft_threads = Vector{Vector{Function}}(undef, ndatasets+1)
        ifft_threads = Vector{Vector{Function}}(undef, ndatasets+1)
        conv_threads = Vector{Vector{Function}}(undef, ndatasets+1)
        corr_threads = Vector{Vector{Function}}(undef, ndatasets+1)
        autocorr_threads = Vector{Vector{Function}}(undef, ndatasets+1)
        fft_threads[1] = [setup_fft(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        ifft_threads[1] = [setup_ifft(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        conv_threads[1] = [setup_conv(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        corr_threads[1] = [setup_corr(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
        autocorr_threads[1] = [setup_autocorr(build_dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]

        nthreads = Threads.nthreads()
        if verb == true
            println("Creating computation buffers for $(nthreads) threads")
        end
        ϵ_threads = zeros(FTYPE, nthreads)
        g_threads_opd = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, nthreads)
        g_threads_ϕ = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ, nthreads)
        g_threads_obj = zeros(FTYPE, build_dim, build_dim, object.nλ, nthreads)
        ϕ_full = zeros(FTYPE, atmosphere.dim, atmosphere.dim, nthreads)
        ϕ_slice = Array{FTYPE, 3}(undef, build_dim, build_dim, nthreads)
        ϕ_composite = zeros(FTYPE, build_dim, build_dim, nthreads)
        o_conv = Array{Function, 3}(undef, object.nλ, patches.npatches, nthreads)
        o_corr = Array{Function, 3}(undef, object.nλ, patches.npatches, nthreads)
        smoothing_kernel = zeros(FTYPE, build_dim, build_dim)
        k_conv = Vector{Function}(undef, nthreads)
        k_corr = Vector{Function}(undef, nthreads)
        P = zeros(Complex{FTYPE}, build_dim, build_dim, nλtotal, nthreads)
        p = zeros(Complex{FTYPE}, build_dim, build_dim, nλtotal, nthreads)
        Î_big = zeros(FTYPE, build_dim, build_dim, nthreads)
        c = zeros(FTYPE, build_dim, build_dim, nthreads)
        d = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
        d2 = zeros(FTYPE, build_dim, build_dim, nthreads)
        containers_builddim_cplx = zeros(Complex{FTYPE}, build_dim, build_dim, nthreads)
        containers_builddim_real = zeros(FTYPE, build_dim, build_dim, nthreads)
        containers_sdim_real = zeros(FTYPE, atmosphere.dim, atmosphere.dim, nthreads)
        containers_sdim_cplx = zeros(Complex{FTYPE}, atmosphere.dim, atmosphere.dim, nthreads)
        ndatasets = length(observations)
        r = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        ω = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        mask_acf = Vector{Matrix{FTYPE}}(undef, ndatasets)
        Î_small = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        containers_pdim_real = Vector{Array{FTYPE, 3}}(undef, ndatasets)
        containers_pdim_cplx = Vector{Array{Complex{FTYPE}, 3}}(undef, ndatasets)

        if verb == true
            println("Creating punch-out and refraction interpolators/adjoints")
        end
        extractor = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλtotal)
        extractor_adj = Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}(undef, ndatasets, observations[1].nepochs, atmosphere.nlayers, nλtotal)
        refraction = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλtotal)
        refraction_adj = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλtotal)
        for dd=1:ndatasets
            fft_threads[dd+1] = [setup_fft(observations[dd].dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
            ifft_threads[dd+1] = [setup_ifft(observations[dd].dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
            conv_threads[dd+1] = [setup_conv(observations[dd].dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
            corr_threads[dd+1] = [setup_corr(observations[dd].dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]
            autocorr_threads[dd+1] = [setup_autocorr(observations[dd].dim, FTYPE=FTYPE) for ~=1:Threads.nthreads()]

            r[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            ω[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            mask_acf[dd] = ones(FTYPE, observations[dd].dim, observations[dd].dim)
            # mask_acf[dd][observations[dd].dim÷2+1 - observations[dd].dim÷4:observations[dd].dim÷2+1 + observations[dd].dim÷4, observations[dd].dim÷2+1 - observations[dd].dim÷4:observations[dd].dim÷2+1 + observations[dd].dim÷4] .= 1
            mask_acf[dd][observations[dd].dim÷2+1, observations[dd].dim÷2+1] = 0
            Î_small[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            containers_pdim_real[dd] = zeros(FTYPE, observations[dd].dim, observations[dd].dim, nthreads)
            containers_pdim_cplx[dd] = zeros(Complex{FTYPE}, observations[dd].dim, observations[dd].dim, nthreads)
            scaleby_wavelength = [observations[dd].detector.λ_nyquist / λtotal[w] for w=1:nλtotal]
            scaleby_height = layer_scale_factors(atmosphere.heights, object.height)            
            Threads.@threads for w=1:nλtotal
                refraction[dd, w] = create_refraction_operator(λtotal[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale, build_dim, FTYPE=FTYPE)
                refraction_adj[dd, w] = create_refraction_adjoint(λtotal[w], atmosphere.λ_ref, observations[dd].ζ, observations[dd].detector.pixscale, build_dim, FTYPE=FTYPE)
                for l=1:atmosphere.nlayers
                    for t=1:observations[dd].nepochs
                        extractor[dd, t, l, w] = create_extractor_operator(atmosphere.positions[:, t, l, w], atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                        extractor_adj[dd, t, l, w] = create_extractor_adjoint(atmosphere.positions[:, t, l, w], atmosphere.dim, build_dim, scaleby_height[l], scaleby_wavelength[w], FTYPE=FTYPE)
                    end
                end
            end
        end

        return new{FTYPE}(extractor, extractor_adj, refraction, refraction_adj, fft_threads, ifft_threads, conv_threads, corr_threads, autocorr_threads, ϕ_full, ϕ_slice, ϕ_composite, o_conv, o_corr, smoothing_kernel, k_conv, k_corr, P, p, c, d, d2, Î_big, Î_small, r, ω, mask_acf, ϵ_threads, g_threads_obj, g_threads_opd, g_threads_ϕ, containers_builddim_real, containers_builddim_cplx, containers_sdim_real, containers_sdim_cplx, containers_pdim_real, containers_pdim_cplx)
    end
end

mutable struct PatchHelpers{T<:AbstractFloat}
    extractor::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    extractor_adj::Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}
    P::Array{Complex{T}, 5}
    p::Array{Complex{T}, 5}
    function PatchHelpers(patches, observations, atmosphere, object; λtotal=atmosphere.λ, build_dim=size(object.object, 1), verb=true, FTYPE=gettype(atmosphere))
        if verb == true
            println("Creating patch extractors and buffers")
        end
        nλtotal = length(λtotal)
        P = zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, nλtotal, Threads.nthreads())
        p = zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, nλtotal, Threads.nthreads())
        
        scaleby_wavelength = atmosphere.λ_nyquist ./ λtotal
        scaleby_height = layer_scale_factors(atmosphere.heights, object.height)
        extractor = create_patch_extractors(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        extractor_adj = create_patch_extractors_adjoint(patches, atmosphere, observations[end], object, scaleby_wavelength, scaleby_height)
        return new{FTYPE}(extractor, extractor_adj, P, p)
    end
end

mutable struct Reconstruction{T<:AbstractFloat}
    λ::Vector{T}
    λtotal::Vector{T}
    nλ::Int64
    nλint::Int64
    nλtotal::Int64
    Δλ::T
    Δλtotal::T
    ndatasets::Int64
    build_dim::Int64
    wavefront_parameter::Symbol
    minimization_scheme::Symbol
    noise_model::Symbol
    weight_function::Function
    fg_object::Function
    gradient_object::Function
    fg_wf::Function
    gradient_wf::Function
    niter_mfbd::Int64
    indx_boot::Vector{UnitRange{Int64}}
    niter_boot::Int64
    maxiter::Int64
    ϵ::T
    grtol::T
    frtol::T
    xrtol::T
    maxeval::Dict{String, Int64}
    regularizers::Regularizers{T}
    smoothing::Bool
    fwhm_schedule::Function
    helpers::Helpers{T}
    patch_helpers::PatchHelpers{T}
    verb_levels::Dict{String, Bool}
    plot::Bool
    figures::ReconstructionFigures
    function Reconstruction(
            atmosphere,
            observations,
            object,
            patches;
            λmin=400.0,
            λmax=1000.0,
            nλ=1,
            nλint=1,
            ndatasets=length(observations),
            build_dim=size(object.object, 1),
            wavefront_parameter=:phase,
            minimization_scheme=:mle,
            noise_model=:gaussian,
            niter_mfbd=10,
            indx_boot=[1:dd for dd=1:ndatasets],
            maxiter=10,
            maxeval=Dict("object"=>100000, "wf"=>100000),
            grtol=1e-9,
            frtol=1e-9,
            xrtol=1e-9,
            smoothing=false,
            maxFWHM=50.0,
            minFWHM=0.5,
            fwhm_schedule=LinearSchedule(maxFWHM, niter_mfbd, minFWHM),
            regularizers=[],
            helpers=[],
            patch_helpers=[],
            verb=true,
            mfbd_verb_level="full",
            plot=true,
            FTYPE = gettype(atmosphere)
        )
        nλtotal = nλ * nλint
        λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
        λtotal = (nλtotal == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλtotal))
        Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
        Δλtotal = (nλtotal == 1) ? 1.0 : (λmax - λmin) / (nλtotal - 1)
        niter_boot = length(indx_boot)
        ϵ = zero(FTYPE)

        weight_function = getfield(Main, Symbol("$(noise_model)_weighting"))
        fg_object = getfield(Main, Symbol("fg_object_$(minimization_scheme)"))
        gradient_object = getfield(Main, Symbol("gradient_object_$(minimization_scheme)_$(noise_model)noise!"))
        fg_wf = getfield(Main, Symbol("fg_$(wavefront_parameter)_$(minimization_scheme)"))
        gradient_wf = getfield(Main, Symbol("gradient_$(wavefront_parameter)_$(minimization_scheme)_$(noise_model)noise!"))
        
        if verb == true
            print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Reconstruction\n"); print(Crayon(reset=true))
            println("\tImage build size: $(build_dim)×$(build_dim) pixels")
            println("\tWavelength: $(minimum(λ))—$(maximum(λ)) nm")
            println("\tNumber of wavelength: $(nλ)")
            println("\tNumber of integrated wavelengths: $(nλint)")
            println("\tNumber of data channels: $(ndatasets)")
            println("\tWavefront Parameter: $(symbol2str[wavefront_parameter])")
            println("\tNoise weighting: $(weight_function)")
            println("\tObject gradient function: $(gradient_object)")
            println("\tWavefront gradient function: $(gradient_wf)")
            println("\tNumber of MFBD cycles: $(niter_mfbd)")
            println("\tMax iterations: $(maxiter)") 
            println("\tMax evaluations: $(maxeval["wf"]) (wf), $(maxeval["object"]) (object)")
            println("\tSmoothing: $(smoothing) (schedule: $(fwhm_schedule), Max FWHM: $(maxFWHM), Min FWHM: $(minFWHM))")
            println("\tStopping criteria: $(grtol) (grtol), $(frtol) (frtol), $(xrtol) (xrtol)")
        end

        if helpers == []
            helpers = Helpers(
                atmosphere, 
                observations,
                object,
                patches,
                λtotal=λtotal,
                verb=verb
            );
        end

        if patch_helpers == []
            patch_helpers = PatchHelpers(
                patches,
                observations,
                atmosphere,
                object,
                λtotal=λtotal,
                verb=verb
            )
        end

        if regularizers == []
            regularizers = Regularizers(verb=verb, FTYPE=FTYPE)
        end

        patches.A = Vector{Array{FTYPE, 6}}(undef, ndatasets)
        patches.phase_slices = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[1].nepochs, atmosphere.nlayers, nλtotal)
        patches.phase_composite = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[1].nepochs, nλtotal)
        patches.psfs = Vector{Array{FTYPE, 6}}(undef, ndatasets)
        for dd=1:ndatasets
            patches.A[dd] = ones(FTYPE, build_dim, build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, nλtotal)
            patches.psfs[dd] = zeros(FTYPE, build_dim, build_dim, patches.npatches, observations[dd].nsubaps, observations[dd].nepochs, nλ)
            observations[dd].model_images = zeros(FTYPE, observations[dd].dim, observations[dd].dim, observations[dd].nsubaps, observations[dd].nepochs)
            observations[dd].w = findall((observations[dd].optics.response .* observations[dd].detector.qe) .> 0)
        end

        if mfbd_verb_level == "full"
            vm = true
            vo = true
        elseif mfbd_verb_level == "mfbd"
            vm = true
            vo = false
        elseif mfbd_verb_level == "silent"
            vm = false
            vo = false
        end
        verb_levels = Dict("vm"=>vm, "vo"=>vo)

        if plot == true
            figs = ReconstructionFigures()
            if wavefront_parameter == :static_phase
                figs.static_phase_fig, figs.static_phase_ax, figs.static_phase_obs = plot_static_phase(observations, show=false)
                display(GLMakie.Screen(), figs.static_phase_fig)
            elseif wavefront_parameter in [:phase, :opd]
                figs.object_fig, figs.object_ax, figs.object_obs = plot_object(object, show=false)
                plot_layers = getfield(Main, Symbol("plot_$(symbol2str[wavefront_parameter])"))
                figs.wf_fig, figs.wf_ax, figs.wf_obs = plot_layers(atmosphere, show=false)
                display(GLMakie.Screen(), figs.object_fig)
                display(GLMakie.Screen(), figs.wf_fig)
            end
        end

        if verb == true
            println("")
        end

        return new{FTYPE}(λ, λtotal, nλ, nλint, nλtotal, Δλ, Δλtotal, ndatasets, build_dim, wavefront_parameter, minimization_scheme, noise_model, weight_function, fg_object, gradient_object, fg_wf, gradient_wf, niter_mfbd, indx_boot, niter_boot, maxiter, ϵ, grtol, frtol, xrtol, maxeval, regularizers, smoothing, fwhm_schedule, helpers, patch_helpers, verb_levels, plot, figs)
    end
end

function gaussian_weighting(entropy, image, rn)
    return 1 / (entropy * rn^2)
end

function mixed_weighting(entropy, image, rn)
    return 1 ./ (entropy .* (image .+ rn^2))
end

function reconstruct!(reconstruction, observations, atmosphere, object, masks, patches; closeplots=true, write=false, folder="", id="")
    FTYPE = gettype(reconstruction)
    for b=1:reconstruction.niter_boot
        current_observations = observations[reconstruction.indx_boot[b]]
        current_masks = masks[reconstruction.indx_boot[b]]
        reconstruction.ϵ = zero(FTYPE)
        for current_iter=1:reconstruction.niter_mfbd
            absolute_iter = (b-1)*reconstruction.niter_mfbd + current_iter        
            update_hyperparams(reconstruction, absolute_iter)
            preconvolve_smoothing(reconstruction)
            preconvolve_object(reconstruction, patches, object)

            if reconstruction.verb_levels["vm"] == true
                print("Bootstrap Iter: $(b) MFBD Iter: $(current_iter) ")
                if reconstruction.smoothing == true
                    print("FWHM: $(reconstruction.fwhm_schedule(absolute_iter)) ")
                end
            end

            ## Reconstruct complex pupil
            if reconstruction.verb_levels["vm"] == true
                print("--> Reconstructing complex pupil ")
            end

            ## Reconstruct Phase
            crit_wf = (x, g) -> reconstruction.fg_wf(x, g, current_observations, atmosphere, current_masks, patches, reconstruction, object)
            vmlmb!(crit_wf, getproperty(atmosphere, reconstruction.wavefront_parameter), verb=reconstruction.verb_levels["vo"], fmin=0, mem=5, maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["wf"])
            if reconstruction.plot == true
                update_layer_figure(atmosphere, reconstruction)
            end

            ## Reconstruct Object
            if reconstruction.verb_levels["vm"] == true
                print("--> object ")
            end

            crit_obj = (x, g) -> reconstruction.fg_object(x, g, current_observations, reconstruction, patches, object)
            vmlmb!(crit_obj, object.object, lower=0, fmin=0, verb=reconstruction.verb_levels["vo"], maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["object"])
            if reconstruction.plot == true
                update_object_figure(dropdims(sum(object.object, dims=3), dims=3), reconstruction)
            end

            ## Compute final criterion
            if reconstruction.verb_levels["vm"] == true
                print("--> ϵ:\t$(reconstruction.ϵ)\n")
            end

            if write == true
                [writefits(observations[dd].model_images, "$(folder)/models_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_recon$(id).fits") for dd=1:reconstruction.ndatasets]
                writefits(object.object, "$(folder)/object_recon$(id).fits")
                writefits(getfield(atmosphere, reconstruction.wavefront_parameter), "$(folder)/$(symbol2str[reconstruction.wavefront_parameter])_recon$(id).fits")
                writefile([reconstruction.ϵ], "$(folder)/recon$(id).dat")
            end
            GC.gc()
        end
    end

    if (reconstruction.plot == true) && (closeplots == true)
        GLMakie.closeall()
    end
end

function reconstruct_static_phase!(reconstruction, observations, atmosphere, object, masks, patches; closeplots=true, write=false, folder="", id="")
    FTYPE = gettype(reconstruction)
    for dd=1:reconstruction.ndatasets
        reconstruction.ϵ = zero(FTYPE)
        for current_iter=1:reconstruction.niter_mfbd
            update_hyperparams(reconstruction, current_iter)
            preconvolve_smoothing(reconstruction)
            preconvolve_object(reconstruction, patches, object)

            if reconstruction.verb_levels["vm"] == true
                print("Channel: $(dd) Iter: $(current_iter) ")
                if reconstruction.smoothing == true
                    print("FWHM: $(reconstruction.fwhm_schedule(current_iter)) ")
                end
            end

            ## Reconstruct complex pupil
            if reconstruction.verb_levels["vm"] == true
                print("--> Reconstructing static phase ")
            end

            ## Reconstruct Phase
            crit_phase_static = (x, g) -> fg_static_phase_mle(x, g, observations, atmosphere, masks, patches, reconstruction)
            vmlmb!(crit_phase_static, observations[dd].phase_static, verb=reconstruction.verb_levels["vo"], fmin=0, mem=5, maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["wf"])
            update_static_phase_figure(atmosphere, reconstruction)

            ## Compute final criterion
            if reconstruction.verb_levels["vm"] == true
                print("--> ϵ:\t$(reconstruction.ϵ)\n")
            end

            if write == true
                writefits(observations[dd].model_images, "$(folder)/static_models_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_recon$(id).fits")
                writefits(observations[dd].static_phase, "$(folder)/static_phase_recon$(id).fits")
                writefile([reconstruction.ϵ], "$(folder)/static_recon$(id).dat")
            end
            GC.gc()
        end
    end

    if (reconstruction.plot == true) && (closeplots == true)
        GLMakie.closeall()
    end
end

@views function height_solve!(observations, atmosphere, object, patches, masks, reconstruction; hmin=ones(atmosphere.nlayers-1), hmax=30.0.*ones(atmosphere.nlayers-1), hstep=ones(atmosphere.nlayers-1), niters=1, verb=true)
    if verb == true
        println("Solving heights for $(atmosphere.nlayers-1) layers")
    end

    FTYPE = gettype(reconstruction)
    nlayers2fit = atmosphere.nlayers - 1
    order2fit = reverse(sortperm(atmosphere.wind[:, 1])[2:end])
    heights = zeros(FTYPE, atmosphere.nlayers)
    heights .= atmosphere.heights
    height_trials = reverse([collect(FTYPE, hmin[l]:hstep[l]:hmax[l]) for l=1:nlayers2fit])
    # [heights[order2fit[l]] = height_trials[l][1] for l=1:nlayers2fit]
    ϵ = [zeros(FTYPE, length(height_trials[l])) for l=1:nlayers2fit]
    [fill!(ϵ[l], FTYPE(NaN)) for l=1:nlayers2fit]
    atmosphere_original = deepcopy(atmosphere)
    object_original = deepcopy(object)

    heights_fig, heights_ax, heights_obs, ϵ_obs = plot_heights(atmosphere, heights=height_trials, ϵ=ϵ, show=true)
    figs = reconstruction.figures
    figs.heights_fig = heights_fig
    figs.heights_ax = heights_ax
    figs.heights_obs = heights_obs
    figs.ϵ_obs = ϵ_obs

    for it=1:niters
        atmosphere = deepcopy(atmosphere_original)
        [fill!(ϵ[l], FTYPE(NaN)) for l=1:nlayers2fit]
        for l=1:nlayers2fit
            for h=1:length(height_trials[l])
                heights[order2fit[l]] = height_trials[l][h]
                print("\tHeight: $(heights)\t")
                change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, reconstruction=reconstruction, verb=false)
                reconstruct!(reconstruction, observations, atmosphere, object, masks, patches, closeplots=false)
                ϵ[l][h] = sum(reconstruction.ϵ[1])
                println("ϵ: $(ϵ[l][h])")
                atmosphere = deepcopy(atmosphere_original)
                object = deepcopy(object_original)
                if reconstruction.plot == true
                    update_heights_figure(height_trials, ϵ, atmosphere, reconstruction)
                end
            end
            heights[order2fit[l]] = height_trials[l][argmin(ϵ[l])]
            change_heights!(patches, atmosphere, object, observations[end], masks[end], heights, reconstruction=reconstruction, verb=false)
        end
        println("Optimal Heights: $(heights)")
        reconstruct!(reconstruction, observations, atmosphere, object, masks, patches, closeplots=false)
        object_original = deepcopy(object)
        atmosphere_original = deepcopy(atmosphere)
        if reconstruction.plot == true
            update_heights_figure(height_trials, ϵ, atmosphere, reconstruction)
        end
    end

    if reconstruction.plot == true
        GLMakie.closeall()
    end

    return ϵ, height_trials, atmosphere, object
end

@views function preconvolve_object(reconstruction, patches, object)
    helpers = reconstruction.helpers
    for w=1:object.nλ
        for np=1:patches.npatches
            for tid=1:Threads.nthreads()
                helpers.o_conv[w, np, tid] = plan_conv_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
                helpers.o_corr[w, np, tid] = plan_ccorr_psf_buffer(object.object[:, :, w], patches.w[:, :, np] .* object.object[:, :, w])[2]
            end
        end
    end
end

@views function preconvolve_smoothing(reconstruction)
    helpers = reconstruction.helpers
    for tid=1:Threads.nthreads()
        helpers.k_conv[tid] = plan_conv_psf_buffer(helpers.smoothing_kernel, helpers.smoothing_kernel)[2]
        helpers.k_corr[tid] = plan_ccorr_psf_buffer(helpers.smoothing_kernel, helpers.smoothing_kernel)[2]
    end
end

function update_hyperparams(reconstruction, iter)
    FTYPE = gettype(reconstruction)
    regularizers = reconstruction.regularizers
    helpers = reconstruction.helpers
    regularizers.βo *= regularizers.βo_schedule(iter)
    regularizers.βwf *= regularizers.βwf_schedule(iter)
    fwhm = reconstruction.fwhm_schedule(iter)
    helpers.smoothing_kernel .= gaussian_kernel(reconstruction.build_dim, fwhm, FTYPE=FTYPE)
end