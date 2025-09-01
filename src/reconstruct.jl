using FFTW
using Statistics
using LinearAlgebra
using OptimPackNextGen


const VERB_LEVELS = Dict(
    :full=>Dict("vm"=>true, "vo"=>true),  # Print output from the MFBD iteration, and the output from OptimPackNextGen
    :reduced=>Dict("vm"=>true, "vo"=>false),  # Print only the output from the MFBD itration
    :silent=>Dict("vm"=>false, "vo"=>false)  # Print nothing
)
const VALID_WAVEFRONT_PARAMETERS = [:psf, :phase, :static_phase, :opd]
const VALID_MINIMIZATION_SCHEMES = [:mle, :mrl]
const VALID_NOISEMODELS = [:gaussian, :mixed]

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

function ExponentialSchedule(maxval, niters, minval)
    α = (minval / maxval)^(1/(niters-1))
    function exponential(iter)
        return max(maxval * α^(iter-1), minval)
    end

    return exponential
end

function ReciprocalSchedule(maxval, minval)
    function reciprocal(x)
        return max(maxval * (1 / x), minval)
    end

    return reciprocal
end

struct BufferChannels{T<:AbstractFloat}
    object_gradient_buffer::Channel{Array{T, 3}}
    wavefront_gradient_buffer::Union{Channel{Array{T, 3}}, Channel{Array{T, 4}}}
    smooth::Union{Channel{Preconvolve{T}}, Channel{Nothing}}
    unsmooth::Union{Channel{Precorrelate{T}}, Channel{Nothing}}
    object_preconv::Channel{Matrix{Preconvolve{T}}}
    object_precorr::Channel{Matrix{Precorrelate{T}}}
    imagedim::Vector{Channel{Matrix{T}}}
    builddim_real::Channel{Matrix{T}}
    builddim_real_4d::Channel{Array{T, 4}}
    builddim_cplx::Channel{Matrix{Complex{T}}}
    builddim_cplx_4d::Channel{Array{Complex{T}, 4}}
    layerdim_real::Channel{Matrix{T}}
    layerdim_cplx::Channel{Matrix{Complex{T}}}
    ft::Channel{Function}
    ift::Channel{Function}
    convolve::Channel{ConvolutionPlan{T}}
    correlate::Channel{CorrelationPlan{T}}
    autocorr::Channel{Function}
    function BufferChannels(
            object, 
            atmosphere, 
            observations,
            patches, 
            build_dim,
            wavefront_parameter,
            frozen_flow,
            smoothing;
            FTYPE=Float64
        )
        nthreads = Threads.nthreads()
        ndatasets = length(observations)

        channel_object_preconv = Channel{Matrix{Preconvolve{FTYPE}}}(nthreads)
        channel_object_precorr = Channel{Matrix{Precorrelate{FTYPE}}}(nthreads)
        foreach(1:nthreads) do ~
            conv_template = Matrix{Preconvolve{FTYPE}}(undef, object.nλ, patches.npatches)
            corr_template = Matrix{Precorrelate{FTYPE}}(undef, object.nλ, patches.npatches)
            fill!(conv_template, Preconvolve(zeros(FTYPE, build_dim, build_dim)))
            fill!(corr_template, Precorrelate(zeros(FTYPE, build_dim, build_dim)))
            put!(channel_object_preconv, conv_template)
            put!(channel_object_precorr, corr_template)
        end

        if smoothing == true
            smoothing_kernel = Matrix{FTYPE}(undef, build_dim, build_dim)
            channel_smooth = Channel{Preconvolve{FTYPE}}(nthreads)
            channel_unsmooth = Channel{Precorrelate{FTYPE}}(nthreads)
            foreach(1:nthreads) do ~
                put!(channel_smooth, Preconvolve(smoothing_kernel))
                put!(channel_unsmooth, Precorrelate(smoothing_kernel))
            end
        else
            smoothing_kernel = FTYPE[;;]
            channel_smooth = Channel{Nothing}(nthreads)
            channel_unsmooth = Channel{Nothing}(nthreads)
            foreach(1:nthreads) do ~
                put!(channel_smooth, nothing)
                put!(channel_unsmooth, nothing)
            end
        end
        
        channel_ft = Channel{Function}(nthreads)
        channel_ift = Channel{Function}(nthreads)
        channel_convolve = Channel{ConvolutionPlan{FTYPE}}(nthreads)
        channel_correlate = Channel{CorrelationPlan{FTYPE}}(nthreads)
        channel_autocorr = Channel{Function}(nthreads)
        foreach(1:nthreads) do ~
            put!(channel_ft, setup_fft(FTYPE, build_dim)[1])
            put!(channel_ift, setup_ifft(Complex{FTYPE}, build_dim)[1])
            put!(channel_convolve, ConvolutionPlan(build_dim, FTYPE=FTYPE))
            put!(channel_correlate, CorrelationPlan(build_dim, FTYPE=FTYPE))
            put!(channel_autocorr, setup_autocorr(FTYPE, build_dim))            
        end

        channel_layerdim_real = Channel{Matrix{FTYPE}}(2*nthreads)
        channel_builddim_cplx_4d = Channel{Array{Complex{FTYPE}, 4}}(2*nthreads)
        channel_builddim_cplx = Channel{Matrix{Complex{FTYPE}}}(2*nthreads)
        foreach(1:2*nthreads) do ~
            put!(channel_layerdim_real, zeros(FTYPE, atmosphere.dim, atmosphere.dim))
            put!(channel_builddim_cplx_4d, zeros(Complex{FTYPE}, build_dim, build_dim, patches.npatches, object.nλ))
            put!(channel_builddim_cplx, zeros(Complex{FTYPE}, build_dim, build_dim))
        end

        if (wavefront_parameter == :phase) && (frozen_flow==true)
            wavefront_zeros = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
            channel_wavefront_gradient_buffer = Channel{Array{FTYPE, 4}}(nthreads)
        elseif (wavefront_parameter == :opd) && (frozen_flow==true)
            wavefront_zeros = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers)
            channel_wavefront_gradient_buffer = Channel{Array{FTYPE, 3}}(nthreads)
        elseif (wavefront_parameter == :phase) && (frozen_flow==false)
            wavefront_zeros = zeros(FTYPE, build_dim, build_dim, observations[1].nepochs, atmosphere.nλ)
            channel_wavefront_gradient_buffer = Channel{Array{FTYPE, 4}}(nthreads)
        elseif (wavefront_parameter == :opd) && (frozen_flow==false)
            wavefront_zeros = zeros(FTYPE, build_dim, build_dim, observations[1].nepochs)
            channel_wavefront_gradient_buffer = Channel{Array{FTYPE, 3}}(nthreads)
        end
        channel_object_gradient_buffer = Channel{Array{FTYPE, 3}}(nthreads)
        channel_layerdim_cplx = Channel{Matrix{Complex{FTYPE}}}(nthreads)
        channel_builddim_real_4d = Channel{Array{FTYPE, 4}}(nthreads)
        foreach(1:nthreads) do ~
            put!(channel_builddim_real_4d, zeros(FTYPE, build_dim, build_dim, patches.npatches, object.nλ))
            put!(channel_layerdim_cplx, zeros(Complex{FTYPE}, atmosphere.dim, atmosphere.dim))
            put!(channel_object_gradient_buffer, zeros(FTYPE, build_dim, build_dim, object.nλ))
            put!(channel_wavefront_gradient_buffer, wavefront_zeros)
        end

        channel_builddim_real = Channel{Matrix{FTYPE}}(10*nthreads)
        foreach(1:10*nthreads) do ~
            put!(channel_builddim_real, zeros(FTYPE, build_dim, build_dim))
        end

        channel_imagedim = Vector{Channel{Matrix{FTYPE}}}(undef, ndatasets)
        for dd=1:ndatasets
            obs = observations[dd]
            channel_imagedim[dd] = Channel{Matrix{FTYPE}}(3*nthreads)
            foreach(1:3*nthreads) do ~
                put!(channel_imagedim[dd], zeros(FTYPE, obs.dim, obs.dim))
            end 
        end

        return new{FTYPE}(channel_object_gradient_buffer, channel_wavefront_gradient_buffer, channel_smooth, channel_unsmooth, channel_object_preconv, channel_object_precorr, channel_imagedim, channel_builddim_real, channel_builddim_real_4d, channel_builddim_cplx, channel_builddim_cplx_4d, channel_layerdim_real, channel_layerdim_cplx, channel_ft, channel_ift, channel_convolve, channel_correlate, channel_autocorr)
    end
end

mutable struct Helpers{T<:AbstractFloat}
    extractor::Vector{Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}}
    extractor_adj::Vector{Array{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}, 4}}
    refraction::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    refraction_adj::Matrix{TwoDimensionalTransformInterpolator{T, LinearSpline{T, Flat}, LinearSpline{T, Flat}}}
    mask_acf::Vector{Matrix{T}}
    smoothing_kernel::Matrix{T}
    channels::BufferChannels{T}
    function Helpers(
            atmosphere,
            observations,
            object,
            patches,
            wavefront_parameter,
            frozen_flow,
            smoothing;
            λtotal=atmosphere.λ,
            build_dim=size(object.object, 1),
            FTYPE=gettype(atmosphere)
        )

        if smoothing == true
            smoothing_kernel = Matrix{FTYPE}(undef, build_dim, build_dim)
        else
            smoothing_kernel = FTYPE[;;]
        end

        nλtotal = length(λtotal)
        ndatasets=length(observations)
        channels = BufferChannels(object, atmosphere, observations, patches, build_dim, wavefront_parameter, frozen_flow, smoothing, FTYPE=FTYPE)
        ndatasets = length(observations)
        mask_acf = Vector{Matrix{FTYPE}}(undef, ndatasets)
        extractor = Vector{Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}}(undef, ndatasets)
        extractor_adj = Vector{Array{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}, 4}}(undef, ndatasets)
        refraction = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλtotal)
        refraction_adj = Matrix{TwoDimensionalTransformInterpolator{FTYPE, LinearSpline{FTYPE, Flat}, LinearSpline{FTYPE, Flat}}}(undef, ndatasets, nλtotal)
        for dd=1:ndatasets
            obs = observations[dd]
            mask_acf[dd] = ones(FTYPE, obs.dim, obs.dim)
            mask_acf[dd][obs.dim÷2+1, obs.dim÷2+1] = 0
            scaleby_wavelength = [atmosphere.λ_nyquist / λtotal[w] for w=1:nλtotal]
            scaleby_height = layer_scale_factors(atmosphere.heights, object.range)
            refraction[dd, :] .= create_refraction_operator.(λtotal, atmosphere.λ_ref, obs.ζ, obs.detector.pixscale, build_dim, FTYPE=FTYPE)
            refraction_adj[dd, :] .= create_refraction_adjoint.(λtotal, atmosphere.λ_ref, obs.ζ, obs.detector.pixscale, build_dim, FTYPE=FTYPE)
            extractor[dd] = create_patch_extractors(patches, atmosphere, obs, object, scaleby_wavelength, scaleby_height, build_dim=build_dim)
            extractor_adj[dd] = create_patch_extractors_adjoint(patches, atmosphere, obs, object, scaleby_wavelength, scaleby_height, build_dim=build_dim)
        end

        return new{FTYPE}(extractor, extractor_adj, refraction, refraction_adj, mask_acf, smoothing_kernel, channels)
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
    fg_wf::Function
    frozen_flow::Bool
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
    minFWHM::T
    maxFWHM::T
    fwhm_schedule::Function
    helpers::Helpers{T}
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
            frozen_flow=true,
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
            fwhm_schedule=ExponentialSchedule(maxFWHM, niter_mfbd, minFWHM),
            regularizers=nothing,
            verb=true,
            mfbd_verb_level=:full,
            plot=true,
            FTYPE = gettype(atmosphere)
        )

        (wavefront_parameter in VALID_WAVEFRONT_PARAMETERS) ? nothing : error("$(wavefront_parameter) not a valid wavefront parameter. Must be one of $(VALID_WAVEFRONT_PARAMETERS)")
        (minimization_scheme in VALID_MINIMIZATION_SCHEMES) ? nothing : error("$(minimization_scheme) not a valid minimization scheme. Must be one of $(VALID_MINIMIZATION_SCHEMES)")
        (noise_model in VALID_NOISEMODELS) ? nothing : error("$(minimization_scheme) not a valid noise model. Must be one of $(VALID_NOISEMODELS)")
        (mfbd_verb_level in keys(VERB_LEVELS)) ? nothing : error("$(mfbd_verb_level) not a valid verbosity level. Must be one of $(keys(VERB_LEVELS))")

        nλtotal = nλ * nλint
        λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
        λtotal = (nλtotal == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλtotal))
        Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
        Δλtotal = (nλtotal == 1) ? 1.0 : (λmax - λmin) / (nλtotal - 1)
        niter_boot = length(indx_boot)
        ϵ = zero(FTYPE)

        weight_function = getfield(Main, Symbol("$(noise_model)_weighting"))
        fg_object = getfield(Main, Symbol("fg_object_$(minimization_scheme)"))
        fg_wf = getfield(Main, Symbol("fg_$(wavefront_parameter)_$(minimization_scheme)"))

        helpers = Helpers(
            atmosphere, 
            observations,
            object,
            patches,
            wavefront_parameter,
            frozen_flow,
            smoothing;
            build_dim=build_dim,
            FTYPE=FTYPE
        );

        if isnothing(regularizers)
            regularizers = Regularizers(verb=verb, FTYPE=FTYPE)
        end

        for dd=1:ndatasets
            obs = observations[dd]
            if frozen_flow == false
                if wavefront_parameter == :phase
                    obs.phase = zeros(FTYPE, build_dim, build_dim, obs.nepochs, nλtotal)
                    # obs.amplitude = zeros(FTYPE, build_dim, build_dim, obs.nepochs, nλtotal)
                elseif wavefront_parameter == :opd
                    obs.opd = zeros(FTYPE, build_dim, build_dim, obs.nepochs)
                end
            end
            obs.model_images = zeros(FTYPE, obs.dim, obs.dim, obs.nsubaps, obs.nepochs)
            obs.w = findall((obs.optics.response .* obs.detector.qe) .> 0)
        end

        verb_levels = VERB_LEVELS[mfbd_verb_level]
        if plot == true
            figs = ReconstructionFigures()
            if wavefront_parameter == :static_phase
                figs.static_phase_fig, figs.static_phase_ax, figs.static_phase_obs = plot_static_phase(observations, show=false)
                Base.display(GLMakie.Screen(), figs.static_phase_fig)
            elseif wavefront_parameter in [:phase, :opd]
                figs.object_fig, figs.object_ax, figs.object_obs = plot_object(object, show=false)
                Base.display(GLMakie.Screen(), figs.object_fig)
                if frozen_flow == true
                    figs.wf_fig, figs.wf_ax, figs.wf_obs = plot_wavefront(observations, atmosphere, wavefront_parameter, show=false)
                    Base.display(GLMakie.Screen(), figs.wf_fig)
                end
            end
            reconstruction = new{FTYPE}(λ, λtotal, nλ, nλint, nλtotal, Δλ, Δλtotal, ndatasets, build_dim, wavefront_parameter, minimization_scheme, noise_model, weight_function, fg_object, fg_wf, frozen_flow, niter_mfbd, indx_boot, niter_boot, maxiter, ϵ, grtol, frtol, xrtol, maxeval, regularizers, smoothing, minFWHM, maxFWHM, fwhm_schedule, helpers, verb_levels, plot, figs)
        else
            reconstruction = new{FTYPE}(λ, λtotal, nλ, nλint, nλtotal, Δλ, Δλtotal, ndatasets, build_dim, wavefront_parameter, minimization_scheme, noise_model, weight_function, fg_object, fg_wf, frozen_flow, niter_mfbd, indx_boot, niter_boot, maxiter, ϵ, grtol, frtol, xrtol, maxeval, regularizers, smoothing, minFWHM, maxFWHM, fwhm_schedule, helpers, verb_levels, plot)
        end

        if verb == true
            display(reconstruction)
        end
        return reconstruction
    end
end

function gaussian_weighting(entropy, image, rn)
    return 1 / (entropy * rn^2)
end

function mixed_weighting(entropy, image, rn)
    return 1 ./ (entropy .* (image .+ rn^2))
end

function reconstruct!(reconstruction, observations, atmosphere, object, patches; closeplots=true, write=false, folder="", id="")
    FTYPE = gettype(reconstruction)
    for b=1:reconstruction.niter_boot
        current_observations = observations[reconstruction.indx_boot[b]]
        ϵ₀ = zero(FTYPE)
        reconstruction.ϵ = zero(FTYPE)
        for current_iter=1:reconstruction.niter_mfbd
            absolute_iter = (b-1)*reconstruction.niter_mfbd + current_iter        
            update_hyperparams(reconstruction, absolute_iter)
            if reconstruction.smoothing == true
                preconvolve_smoothing(reconstruction)
            end
            preconvolve_object(reconstruction, patches, object)

            if reconstruction.verb_levels["vm"] == true
                print("Bootstrap Iter: $(b)/$(reconstruction.niter_boot)\tMFBD Iter: $(current_iter)/$(reconstruction.niter_mfbd) ")
                if reconstruction.smoothing == true
                    print("\tFWHM:$(reconstruction.fwhm_schedule(absolute_iter)) ")
                end
            end

            ## Reconstruct complex pupil
            if reconstruction.verb_levels["vm"] == true
                print("--> Reconstructing complex pupil ")
                if reconstruction.verb_levels["vo"] == true
                    println()
                end
            end

            if reconstruction.frozen_flow == true
                starting_wf = getproperty(atmosphere, reconstruction.wavefront_parameter)
            else
                starting_wf = getproperty(observations[1], reconstruction.wavefront_parameter)
            end

            ## Reconstruct Phase
            crit_wf = (x, g) -> reconstruction.fg_wf(x, g, current_observations, atmosphere, patches, reconstruction, object)
            vmlmb!(crit_wf, starting_wf, verb=reconstruction.verb_levels["vo"], fmin=0, mem=5, maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["wf"])
            if (reconstruction.plot == true) && (reconstruction.frozen_flow == true)
                update_layer_figure(atmosphere, reconstruction)
            end

            ## Reconstruct Object
            if reconstruction.verb_levels["vm"] == true
                print("--> object ")
                if reconstruction.verb_levels["vo"] == true
                    println()
                end
            end

            crit_obj = (x, g) -> reconstruction.fg_object(x, g,  current_observations, atmosphere, patches, reconstruction, object)
            vmlmb!(crit_obj, object.object, lower=0, fmin=0, verb=reconstruction.verb_levels["vo"], maxiter=reconstruction.maxiter, gtol=(0, reconstruction.grtol), ftol=(0, reconstruction.frtol), xtol=(0, reconstruction.xrtol), maxeval=reconstruction.maxeval["object"])
            if reconstruction.plot == true
                update_object_figure(dropdims(sum(object.object, dims=3), dims=3), reconstruction)
            end

            ## Compute final criterion
            if reconstruction.verb_levels["vm"] == true
                println("--> ϵ:\t$(reconstruction.ϵ)")
            end

            if write == true
                [writefits(observations[dd].model_images, "$(folder)/models_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_recon$(id).fits") for dd=1:reconstruction.ndatasets]
                writefits(object.object, "$(folder)/object_recon$(id).fits")
                if reconstruction.frozen_flow == true
                    writefits(getfield(atmosphere, reconstruction.wavefront_parameter), "$(folder)/$(symbol2str[reconstruction.wavefront_parameter])_recon$(id).fits")
                    writefile([reconstruction.ϵ], "$(folder)/recon$(id).dat")
                else
                    [writefits(observations[dd].phase, "$(folder)/phase_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_recon$(id).fits") for dd=1:reconstruction.ndatasets]
                end
            end

            if reconstruction.ϵ == ϵ₀
                break
            end
            ϵ₀ = reconstruction.ϵ
            GC.gc()
        end
    end

    if (reconstruction.plot == true) && (closeplots == true)
        GLMakie.closeall()
    end
end

@views function height_solve!(observations, atmosphere, object, patches, masks, reconstruction; hmin=1000.0.*ones(atmosphere.nlayers-1), hmax=30000.0.*ones(atmosphere.nlayers-1), hstep=1000.0.*ones(atmosphere.nlayers-1), niters=1, verb=true)
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
                change_heights!(patches, atmosphere, object, observations, masks, heights, reconstruction=reconstruction, verb=false)
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
            change_heights!(patches, atmosphere, object, observations, masks, heights, reconstruction=reconstruction, verb=false)
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
    foreach(1:Threads.nthreads()) do ~
        object_preconv = take!(helpers.channels.object_preconv)
        object_precorr = take!(helpers.channels.object_precorr)
        for w=1:object.nλ
            for np=1:patches.npatches
                object_patch = patches.w[:, :, np] .* object.object[:, :, w]
                object_preconv[w, np] = Preconvolve(object_patch)
                object_precorr[w, np] = Precorrelate(object_patch)
            end
        end
        put!(helpers.channels.object_preconv, object_preconv)
        put!(helpers.channels.object_precorr, object_precorr)
    end
end

@views function preconvolve_smoothing(reconstruction)
    helpers = reconstruction.helpers
    channel_smooth = helpers.channels.smooth
    channel_unsmooth = helpers.channels.unsmooth
    foreach(1:Threads.nthreads()) do ~
        take!(channel_smooth)
        take!(channel_unsmooth)
        put!(channel_smooth, Preconvolve(fftshift(helpers.smoothing_kernel)))
        put!(channel_unsmooth, Precorrelate(fftshift(helpers.smoothing_kernel)))
    end
end

function update_hyperparams(reconstruction, iter)
    FTYPE = gettype(reconstruction)
    regularizers = reconstruction.regularizers
    helpers = reconstruction.helpers
    regularizers.βo *= regularizers.βo_schedule(iter)
    regularizers.βwf *= regularizers.βwf_schedule(iter)
    fwhm = reconstruction.fwhm_schedule(iter)
    if reconstruction.smoothing == true
        helpers.smoothing_kernel .= gaussian_kernel(reconstruction.build_dim, fwhm, FTYPE=FTYPE)
    end
end
