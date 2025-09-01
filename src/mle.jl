function loglikelihood_gaussian(data, model, ω)
    FTYPE = eltype(data)
    r = zeros(FTYPE, size(data))
    ϵ = FTYPE(loglikelihood_gaussian!(r, data, model, ω))
    return ϵ
end

function loglikelihood_gaussian!(r, data, model, ω)
    FTYPE = eltype(r)
    r .= model .- data
    ϵ = FTYPE(mapreduce(x -> x[1] * x[2]^2, +, zip(ω, r)))
    return ϵ
end

@views function fg_object_mle(x, g, observations, atmosphere, patches, reconstruction, object)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    regularizers = reconstruction.regularizers
    zeros!(g)
    reconstruction.ϵ = zero(FTYPE)
    if reconstruction.plot == true
        update_object_figure(dropdims(mean(x, dims=3), dims=3), reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        obs = observations[dd]  # Dataset from the full set
        detector = obs.detector
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
        ##
        zeros!(obs.model_images)  # Fill the model images with zeros to ensure a fresh start
        reconstruction.ϵ += tmapreduce(+, collect(Iterators.product(1:obs.nepochs, 1:obs.nsubaps))) do (t, n)  # Loop through all timesteps
            extractors = helpers.extractor[dd][t, :, :, :]  # Interpolation operators for punch out
            ϵ_local = zero(FTYPE)
            buffers = take_object_buffers(helpers, dd)
            subap_image = obs.model_images[:, :, n, t]  # Model image for each subap at each time
            ##
            create_radiant_energy_pre_detector!(subap_image, obs, object, atmosphere, patches, refraction, extractors, buffers, (; n, t), frozen_flow=reconstruction.frozen_flow)
            subap_image ./= detector.gain
            buffers.ω .= reconstruction.weight_function(obs.entropy[n, t], subap_image, detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
            ϵ_local += loglikelihood_gaussian!(buffers.r, obs.images[:, :, n, t], subap_image, buffers.ω)  # Calculate the gaussian likelihood for the calculated model image and data frame
            gradient_object_mle!(reconstruction, obs, atmosphere, patches, buffers, (; n, t))
            put_object_buffers(helpers, dd, buffers)

            return ϵ_local
        end
    end

    for ~=1:Threads.nthreads()
        buffer = take!(helpers.channels.object_gradient_buffer)
        g .+= buffer
        zeros!(buffer)
        put!(helpers.channels.object_gradient_buffer, buffer)
    end

    for w=1:reconstruction.nλ
        reconstruction.ϵ += regularizers.o_reg(x[:, :, w], g[:, :, w], regularizers.βo)
    end
    reconstruction.ϵ += regularizers.λ_reg(x, g, regularizers.βλ)

    return reconstruction.ϵ
end

@views function gradient_object_mle!(reconstruction, obs, atmosphere, patches, buffers, ixs)
    if reconstruction.noise_model == :gaussian
        buffers.r .*= 2 .* buffers.ω
    elseif reconstruction.noise_model == :mixed
        buffers.r .= 2 .* buffers.ω .* buffers.r .- (buffers.ω .* buffers.r).^2 .* obs.entropy[ixs.n, ixs.t]
    end

    block_replicate!(buffers.image_big, buffers.r)
    for np=1:patches.npatches
        for w₁=1:reconstruction.nλ
            correlate!(buffers.container_builddim_real, buffers.corr_plan, buffers.image_big, buffers.psf[:, :, np, w₁])
            buffers.gradient_buffer[:, :, w₁] .+= (reconstruction.Δλ*obs.optics.response[w₁]*atmosphere.transmission[w₁]*obs.detector.exptime*obs.aperture_area/obs.detector.gain) .* patches.w[:, :, np] .* buffers.container_builddim_real
        end
    end
end

@views function fg_phase_mle(x, g, observations, atmosphere, patches, reconstruction, object)
    FTYPE = gettype(reconstruction)  # Alias for the datatype
    if reconstruction.frozen_flow == true
        setfield!(atmosphere, reconstruction.wavefront_parameter, x)
        if reconstruction.plot == true  # If plotting is enabled, object and phase plots will be updated here with the current proposed values
            update_phase_figure(x, atmosphere, reconstruction)
        end
    end
    ndatasets = length(observations)  # Number of datasets to be processed
    helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
    regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
    zeros!(g)  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values 
    reconstruction.ϵ = zero(FTYPE)
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        obs = observations[dd]  # Dataset from the full set
        if reconstruction.frozen_flow == false
            setfield!(obs, reconstruction.wavefront_parameter, x)
        end
        detector = obs.detector
        extractors = helpers.extractor[dd]
        extractors_adj = helpers.extractor_adj[dd]  # Interpolation operators for punch out        
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
        refraction_adj = helpers.refraction_adj[dd, :]  # Inverse refraction operator for the dataset
        zeros!(obs.model_images)
        ##
        reconstruction.ϵ += tmapreduce(+, collect(Iterators.product(1:obs.nepochs, 1:obs.nsubaps))) do (t, n)  # Loop through all timesteps
            ϵ_local = zero(FTYPE)
            buffers = take_wf_buffers(helpers, dd)
            subap_image = obs.model_images[:, :, n, t]  # Model image for each subap at each time
            create_radiant_energy_pre_detector!(subap_image, obs, object, atmosphere, patches, refraction, extractors[t, :, :, :], buffers, (; n, t), frozen_flow=reconstruction.frozen_flow)
            subap_image ./= detector.gain
            buffers.ω .= reconstruction.weight_function(obs.entropy[n, t], subap_image, detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
            ϵ_local += loglikelihood_gaussian!(buffers.r, obs.images[:, :, n, t], subap_image, buffers.ω)  # Calculate the gaussian likelihood for the calculated model image and data frame
            gradient_phase_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractors_adj[t, :, :, :], buffers, (; n, t))
            put_wf_buffers(helpers, dd, buffers)
            return ϵ_local
        end
    end

    for ~=1:Threads.nthreads()
        buffer = take!(helpers.channels.wavefront_gradient_buffer)
        g .+= buffer
        zeros!(buffer)
        put!(helpers.channels.wavefront_gradient_buffer, buffer)
    end

    ## Apply regularization
    if reconstruction.frozen_flow == true
        for l=1:atmosphere.nlayers
            for w₁=1:reconstruction.nλ
                for w₂=1:reconstruction.nλint 
                    w = (w₁-1)*reconstruction.nλint + w₂
                    reconstruction.ϵ += regularizers.wf_reg(x[:, :, l, w], g[:, :, l, w], regularizers.βwf)
                end
            end
        end
    else
        for t=1:observations[1].nepochs
            for w₁=1:reconstruction.nλ
                for w₂=1:reconstruction.nλint 
                    w = (w₁-1)*reconstruction.nλint + w₂
                    reconstruction.ϵ += regularizers.wf_reg(x[:, :, t, w], g[:, :, t, w], regularizers.βwf)
                end
            end
        end
    end

    return reconstruction.ϵ
end

@views function gradient_phase_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractor_adj, buffers, ixs)
    if reconstruction.noise_model == :gaussian
        buffers.r .*= 2 .* buffers.ω
    elseif reconstruction.noise_model == :mixed
        buffers.r .= 2 .* buffers.ω .* buffers.r .- (buffers.ω .* buffers.r).^2 .* obs.entropy[ixs.n, ixs.t]
    end

    if reconstruction.frozen_flow == true
        gradient_phase_ffm_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractor_adj, buffers)
    else
        gradient_phase_nonffm_mle!(reconstruction, obs, atmosphere, refraction_adj, buffers, ixs)
    end
end

@views function gradient_phase_ffm_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractor_adj, buffers)
    block_replicate!(buffers.c, buffers.r)
    conj!(buffers.p)
    for np=1:patches.npatches
        for w₁=1:reconstruction.nλ
            correlate!(buffers.d2, buffers.object_precorr[w₁, np], buffers.c)            
            for w₂=1:reconstruction.nλint
                w = (w₁-1)*reconstruction.nλint + w₂
                buffers.container_builddim_real .= buffers.d2
                mul!(buffers.d2, refraction_adj[w], buffers.container_builddim_real)

                buffers.p[:, :, np, w] .*= buffers.d2
                buffers.iffts(buffers.d, buffers.p[:, :, np, w])

                buffers.d .*= buffers.P[:, :, np, w]
                buffers.d2 .= imag.(buffers.d)
                buffers.d2 .*= -2 * obs.optics.response[w] * atmosphere.transmission[w] * obs.detector.gain * obs.detector.exptime * obs.aperture_area

                if !isnothing(buffers.unsmooth)
                    correlate!(buffers.d2, buffers.unsmooth, buffers.d2)
                end

                for l=1:atmosphere.nlayers
                    mul!(buffers.container_layerdim_real, extractor_adj[np, l, w], buffers.d2)
                    buffers.gradient_buffer[:, :, l, w] .+= buffers.container_layerdim_real
                end
            end
        end
    end
end

@views function gradient_phase_nonffm_mle!(reconstruction, obs, atmosphere, refraction_adj, buffers, ixs)
    block_replicate!(buffers.c, buffers.r)
    conj!(buffers.p)
    for w₁=1:reconstruction.nλ
        correlate!(buffers.d2, buffers.object_precorr[w₁, 1], buffers.c)
        for w₂=1:reconstruction.nλint
            w = (w₁-1)*reconstruction.nλint + w₂
            buffers.container_builddim_real .= buffers.d2
            mul!(buffers.d2, refraction_adj[w], buffers.container_builddim_real)

            buffers.p[:, :, 1, w] .*= buffers.d2
            buffers.iffts(buffers.d, buffers.p[:, :, 1, w])

            buffers.d .*= buffers.P[:, :, 1, w]
            buffers.d2 .= imag.(buffers.d)
            buffers.d2 .*= -2 * obs.optics.response[w] * atmosphere.transmission[w] * obs.detector.gain * obs.detector.exptime * obs.aperture_area

            if !isnothing(buffers.unsmooth)
                correlate!(buffers.d2, buffers.unsmooth, buffers.d2)
            end

            buffers.gradient_buffer[:, :, ixs.t, w] .+= buffers.d2
        end
    end
end

@views function fg_opd_mle(x, g, observations, atmosphere, patches, reconstruction, object)
    FTYPE = gettype(reconstruction)  # Alias for the datatype
    if reconstruction.frozen_flow == true
        setfield!(atmosphere, reconstruction.wavefront_parameter, x)
        if reconstruction.plot == true  # If plotting is enabled, object and phase plots will be updated here with the current proposed values
            update_phase_figure(x, atmosphere, reconstruction)
        end
    end

    if reconstruction.frozen_flow == true
        for w=1:reconstruction.nλ
            atmosphere.phase[:, :, :, w] .= x .* (2pi / reconstruction.λ[w])
        end
    end

    ndatasets = length(observations)  # Number of datasets to be processed
    helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
    regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
    zeros!(g)  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values 
    reconstruction.ϵ = zero(FTYPE)
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        obs = observations[dd]  # Dataset from the full set
        if reconstruction.frozen_flow == false
            setfield!(obs, reconstruction.wavefront_parameter, x)
            for w=1:reconstruction.nλ
                obs.phase[:, :, :, w] .= x .* (2pi / reconstruction.λ[w])
            end
        end
        detector = obs.detector
        extractors = helpers.extractor[dd]
        extractors_adj = helpers.extractor_adj[dd]  # Interpolation operators for punch out        
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
        refraction_adj = helpers.refraction_adj[dd, :]  # Inverse refraction operator for the dataset
        zeros!(obs.model_images)
        ##
        reconstruction.ϵ += tmapreduce(+, collect(Iterators.product(1:obs.nepochs, 1:obs.nsubaps))) do (t, n)  # Loop through all timesteps
            ϵ_local = zero(FTYPE)
            buffers = take_wf_buffers(helpers, dd)
            subap_image = obs.model_images[:, :, n, t]  # Model image for each subap at each time
            create_radiant_energy_pre_detector!(subap_image, obs, object, atmosphere, patches, refraction, extractors[t, :, :, :], buffers, (; n, t), frozen_flow=reconstruction.frozen_flow)
            subap_image ./= detector.gain
            buffers.ω .= reconstruction.weight_function(obs.entropy[n, t], subap_image, detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
            ϵ_local += loglikelihood_gaussian!(buffers.r, obs.images[:, :, n, t], subap_image, buffers.ω)  # Calculate the gaussian likelihood for the calculated model image and data frame
            gradient_opd_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractors_adj[t, :, :, :], buffers, (; n, t))
            put_wf_buffers(helpers, dd, buffers)
            return ϵ_local
        end
    end

    for ~=1:Threads.nthreads()
        buffer = take!(helpers.channels.wavefront_gradient_buffer)
        g .+= buffer
        zeros!(buffer)
        put!(helpers.channels.wavefront_gradient_buffer, buffer)
    end

    ## Apply regularization
    if reconstruction.frozen_flow == true
        for l=1:atmosphere.nlayers
            for w₁=1:reconstruction.nλ
                for w₂=1:reconstruction.nλint 
                    w = (w₁-1)*reconstruction.nλint + w₂
                    reconstruction.ϵ += regularizers.wf_reg(x[:, :, l, w], g[:, :, l, w], regularizers.βwf)
                end
            end
        end
    else
        for t=1:observations[1].nepochs
            for w₁=1:reconstruction.nλ
                for w₂=1:reconstruction.nλint 
                    w = (w₁-1)*reconstruction.nλint + w₂
                    reconstruction.ϵ += regularizers.wf_reg(x[:, :, t, w], g[:, :, t, w], regularizers.βwf)
                end
            end
        end
    end

    return reconstruction.ϵ
end

@views function gradient_opd_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractor_adj, buffers, ixs)
    if reconstruction.noise_model == :gaussian
        buffers.r .*= 2 .* buffers.ω
    elseif reconstruction.noise_model == :mixed
        buffers.r .= 2 .* buffers.ω .* buffers.r .- (buffers.ω .* buffers.r).^2 .* obs.entropy[ixs.n, ixs.t]
    end

    if reconstruction.frozen_flow == true
        gradient_opd_ffm_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractor_adj, buffers)
    else
        gradient_opd_nonffm_mle!(reconstruction, obs, atmosphere, refraction_adj, buffers, ixs)
    end
end

@views function gradient_opd_ffm_mle!(reconstruction, obs, atmosphere, patches, refraction_adj, extractor_adj, buffers)
    FTYPE = gettype(reconstruction)
    block_replicate!(buffers.c, buffers.r)
    conj!(buffers.p)
    for np=1:patches.npatches
        for w₁=1:reconstruction.nλ
            correlate!(buffers.d2, buffers.object_precorr[w₁, np], buffers.c)            
            for w₂=1:reconstruction.nλint
                w = (w₁-1)*reconstruction.nλint + w₂
                buffers.container_builddim_real .= buffers.d2
                mul!(buffers.d2, refraction_adj[w], buffers.container_builddim_real)

                buffers.p[:, :, np, w] .*= buffers.d2
                buffers.iffts(buffers.d, buffers.p[:, :, np, w])

                buffers.d .*= buffers.P[:, :, np, w]
                buffers.d2 .= imag.(buffers.d)
                buffers.d2 .*= FTYPE(-4pi) * reconstruction.Δλ/reconstruction.λ[w] * obs.optics.response[w] * atmosphere.transmission[w] * obs.detector.gain * obs.detector.exptime * obs.aperture_area

                if !isnothing(buffers.unsmooth)
                    correlate!(buffers.d2, buffers.unsmooth, buffers.d2)
                end

                for l=1:atmosphere.nlayers
                    mul!(buffers.container_layerdim_real, extractor_adj[np, l, w], buffers.d2)
                    buffers.gradient_buffer[:, :, l] .+= buffers.container_layerdim_real
                end
            end
        end
    end
end

@views function gradient_opd_nonffm_mle!(reconstruction, obs, atmosphere, refraction_adj, buffers, ixs)
    FTYPE = gettype(reconstruction)
    block_replicate!(buffers.c, buffers.r)
    conj!(buffers.p)
    for w₁=1:reconstruction.nλ
        correlate!(buffers.d2, buffers.object_precorr[w₁, 1], buffers.c)
        for w₂=1:reconstruction.nλint
            w = (w₁-1)*reconstruction.nλint + w₂
            buffers.container_builddim_real .= buffers.d2
            mul!(buffers.d2, refraction_adj[w], buffers.container_builddim_real)

            buffers.p[:, :, 1, w] .*= buffers.d2
            buffers.iffts(buffers.d, buffers.p[:, :, 1, w])

            buffers.d .*= buffers.P[:, :, 1, w]
            buffers.d2 .= imag.(buffers.d)
            buffers.d2 .*= FTYPE(-4pi) * reconstruction.Δλ/reconstruction.λ[w] * obs.optics.response[w] * atmosphere.transmission[w] * obs.detector.gain * obs.detector.exptime * obs.aperture_area

            if !isnothing(buffers.unsmooth)
                correlate!(buffers.d2, buffers.unsmooth, buffers.d2)
            end

            buffers.gradient_buffer[:, :, ixs.t] .+= buffers.d2
        end
    end
end

# @views function fg_psf_mle(x, g, observations, atmosphere, masks, patches, reconstruction, object)
# end

# @views function gradient_psf_mle_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, precorr_object, entropy, unsmooth!, refraction_adj, ifft!, container_builddim_real)
# end

# @views function gradient_psf_mle_mixednoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, precorr_object, entropy, unsmooth!, refraction_adj, ifft!, container_builddim_real)
# end

# @views function gradient_psf_mle!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, precorr_object, entropy, unsmooth!, refraction_adj, ifft!, container_builddim_real)
# end


function take_object_buffers(helpers, dd)
    iffts = take!(helpers.channels.ift)  # Pre-allocated FFTs
    conv_plan = take!(helpers.channels.convolve)  # Pre-allocated convolutions
    corr_plan = take!(helpers.channels.correlate)  # Pre-allocated correlations
    A = take!(helpers.channels.builddim_real)  # Buffer for amplitude
    ϕ_slices = take!(helpers.channels.builddim_real)  # Buffer for per-layer phase
    ϕ_composite = take!(helpers.channels.builddim_real)  # Buffer for composite phase
    smooth = take!(helpers.channels.smooth)  # Function to smooth the composite phase
    P = take!(helpers.channels.builddim_cplx_4d)  # Pupil function buffers 
    p = take!(helpers.channels.builddim_cplx_4d)  # IFFT of pupil function buffers
    psf = take!(helpers.channels.builddim_real_4d)  # PSF buffer
    psf_temp = take!(helpers.channels.builddim_real)  # Temporary array needed to compute PSF
    object_patch = take!(helpers.channels.builddim_real)  # Object-times-patch weight buffer
    ω = take!(helpers.channels.imagedim[dd])
    r = take!(helpers.channels.imagedim[dd])
    image_big = take!(helpers.channels.builddim_real)  # Buffer to hold the full-size spectral image
    image_small = take!(helpers.channels.imagedim[dd])  # Buffer to hold the downsampled spectral image
    container_builddim_real = take!(helpers.channels.builddim_real)
    gradient_buffer = take!(helpers.channels.object_gradient_buffer)
    ones!(A)
    zeros!(psf)
    return (; iffts, conv_plan, corr_plan, A, ϕ_slices, ϕ_composite, smooth, P, p, psf, psf_temp, object_patch, ω, r, image_big, image_small, container_builddim_real, gradient_buffer)
end

function put_object_buffers(helpers, dd, buffers)
    put!(helpers.channels.ift, buffers.iffts)  # Pre-allocated FFTs
    put!(helpers.channels.convolve, buffers.conv_plan)  # Pre-allocated convolutions
    put!(helpers.channels.correlate, buffers.corr_plan)  # Pre-allocated correlations
    put!(helpers.channels.builddim_real, buffers.A)  # Buffer for amplitude
    put!(helpers.channels.builddim_real, buffers.ϕ_slices)  # Buffer for per-layer phase
    put!(helpers.channels.builddim_real, buffers.ϕ_composite)  # Buffer for composite phase
    put!(helpers.channels.smooth, buffers.smooth)  # Function to smooth the composite phase
    put!(helpers.channels.builddim_cplx_4d, buffers.P)  # Pupil function buffers 
    put!(helpers.channels.builddim_cplx_4d, buffers.p)  # IFFT of pupil function buffers
    put!(helpers.channels.builddim_real_4d, buffers.psf)  # PSF buffer
    put!(helpers.channels.builddim_real, buffers.psf_temp)  # Temporary array needed to compute PSF
    put!(helpers.channels.builddim_real, buffers.object_patch)  # Object-times-patch weight buffer
    put!(helpers.channels.imagedim[dd], buffers.ω)
    put!(helpers.channels.imagedim[dd], buffers.r)
    put!(helpers.channels.builddim_real, buffers.image_big)  # Buffer to hold the full-size spectral image
    put!(helpers.channels.imagedim[dd], buffers.image_small)  # Buffer to hold the downsampled spectral image
    put!(helpers.channels.builddim_real, buffers.container_builddim_real)
    put!(helpers.channels.object_gradient_buffer, buffers.gradient_buffer)
end

function take_wf_buffers(helpers, dd)
    iffts = take!(helpers.channels.ift)  # Pre-allocated FFTs
    conv_plan = take!(helpers.channels.convolve)  # Pre-allocated convolutions
    corr_plan = take!(helpers.channels.correlate)  # Pre-allocated correlations
    A = take!(helpers.channels.builddim_real)  # Buffer for amplitude
    ϕ_slices = take!(helpers.channels.builddim_real)  # Buffer for per-layer phase
    ϕ_composite = take!(helpers.channels.builddim_real)  # Buffer for composite phase
    smooth = take!(helpers.channels.smooth)  # Function to smooth the composite phase
    unsmooth = take!(helpers.channels.unsmooth)  # Function to smooth the composite phase
    P = take!(helpers.channels.builddim_cplx_4d)  # Pupil function buffers 
    p = take!(helpers.channels.builddim_cplx_4d)  # IFFT of pupil function buffers
    psf = take!(helpers.channels.builddim_real_4d)  # PSF buffer
    psf_temp = take!(helpers.channels.builddim_real)  # Temporary array needed to compute PSF
    object_patch = take!(helpers.channels.builddim_real)  # Object-times-patch weight buffer
    object_precorr = take!(helpers.channels.object_precorr)
    ω = take!(helpers.channels.imagedim[dd])
    r = take!(helpers.channels.imagedim[dd])
    image_big = take!(helpers.channels.builddim_real)  # Buffer to hold the full-size spectral image
    image_small = take!(helpers.channels.imagedim[dd])  # Buffer to hold the downsampled spectral image
    c = take!(helpers.channels.builddim_real)
    d = take!(helpers.channels.builddim_cplx)
    d2 = take!(helpers.channels.builddim_real)
    container_builddim_real = take!(helpers.channels.builddim_real)
    container_layerdim_real = take!(helpers.channels.layerdim_real)
    gradient_buffer = take!(helpers.channels.wavefront_gradient_buffer)
    ones!(A)
    zeros!(psf)
    return (; iffts, conv_plan, corr_plan, A, ϕ_slices, ϕ_composite, smooth, unsmooth, P, p, psf, psf_temp, object_patch, object_precorr, ω, r, image_big, image_small, c, d, d2, container_builddim_real, container_layerdim_real, gradient_buffer)
end

function put_wf_buffers(helpers, dd, buffers)
    put!(helpers.channels.ift, buffers.iffts)  # Pre-allocated FFTs
    put!(helpers.channels.convolve, buffers.conv_plan)  # Pre-allocated convolutions
    put!(helpers.channels.correlate, buffers.corr_plan)  # Pre-allocated correlations
    put!(helpers.channels.builddim_real, buffers.A)  # Buffer for amplitude
    put!(helpers.channels.builddim_real, buffers.ϕ_slices)  # Buffer for per-layer phase
    put!(helpers.channels.builddim_real, buffers.ϕ_composite)  # Buffer for composite phase
    put!(helpers.channels.smooth, buffers.smooth)  # Function to smooth the composite phase
    put!(helpers.channels.unsmooth, buffers.unsmooth)  # Function to smooth the composite phase
    put!(helpers.channels.builddim_cplx_4d, buffers.P)  # Pupil function buffers 
    put!(helpers.channels.builddim_cplx_4d, buffers.p)  # IFFT of pupil function buffers
    put!(helpers.channels.builddim_real_4d, buffers.psf)  # PSF buffer
    put!(helpers.channels.builddim_real, buffers.psf_temp)  # Temporary array needed to compute PSF
    put!(helpers.channels.builddim_real, buffers.object_patch)  # Object-times-patch weight buffer
    put!(helpers.channels.object_precorr, buffers.object_precorr)
    put!(helpers.channels.imagedim[dd], buffers.ω)
    put!(helpers.channels.imagedim[dd], buffers.r)
    put!(helpers.channels.builddim_real, buffers.image_big)  # Buffer to hold the full-size spectral image
    put!(helpers.channels.imagedim[dd], buffers.image_small)  # Buffer to hold the downsampled spectral image
    put!(helpers.channels.builddim_real, buffers.c)
    put!(helpers.channels.builddim_cplx, buffers.d)
    put!(helpers.channels.builddim_real, buffers.d2)
    put!(helpers.channels.layerdim_real, buffers.container_layerdim_real)
    put!(helpers.channels.builddim_real, buffers.container_builddim_real)
    put!(helpers.channels.wavefront_gradient_buffer, buffers.gradient_buffer)
end
