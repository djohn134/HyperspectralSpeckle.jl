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
    fill!(g, zero(FTYPE))
    reconstruction.ϵ = zero(FTYPE)
    if reconstruction.plot == true
        update_object_figure(dropdims(mean(x, dims=3), dims=3), reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        obs = observations[dd]  # Dataset from the full set
        detector = obs.detector
        optics = obs.optics  # Makes it easier to type
        mask = obs.masks  # Masks for that dataset
        scale_psfs = mask.scale_psfs  # Scaler to multiply the PSFs by to ensure unit volume
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
        ϕ_static = obs.phase_static  # Static phase for the dataset
        ##
        fill!(obs.model_images, zero(FTYPE))  # Fill the model images with zeros to ensure a fresh start
        reconstruction.ϵ += tmapreduce(+, 1:obs.nepochs) do t  # Loop through all timesteps
            ϵ_local = zero(FTYPE)
            extractor = helpers.extractor[dd][t, :, :, :]  # Interpolation operators for punch out
            ## Aliases for time-dependent parameters
            iffts = take!(helpers.channel_ift)  # Pre-allocated FFTs
            convs = take!(helpers.channel_convolve)  # Pre-allocated convolutions
            corrs = take!(helpers.channel_correlate)  # Pre-allocated correlations
            A = take!(helpers.channel_builddim_real)  # Buffer for amplitude
            ϕ_slices = take!(helpers.channel_builddim_real)  # Buffer for per-layer phase
            ϕ_composite = take!(helpers.channel_builddim_real)  # Buffer for composite phase
            smoothing = take!(helpers.channel_smooth)  # Function to smooth the composite phase
            P = take!(helpers.channel_builddim_cplx_4d)  # Pupil function buffers 
            p = take!(helpers.channel_builddim_cplx_4d)  # IFFT of pupil function buffers
            psf = take!(helpers.channel_builddim_real_4d)  # PSF buffer
            psf_temp = take!(helpers.channel_builddim_real)  # Temporary array needed to compute PSF
            object_patch = take!(helpers.channel_builddim_real)  # Object-times-patch weight buffer
            ω = take!(helpers.channel_imagedim[dd])
            r = take!(helpers.channel_imagedim[dd])
            image_big = take!(helpers.channel_builddim_real)  # Buffer to hold the full-size spectral image
            image_small = take!(helpers.channel_imagedim[dd])  # Buffer to hold the downsampled spectral image
            container_builddim_real = take!(helpers.channel_builddim_real)
            gradient_buffer = take!(helpers.channel_object_gradient_buffer)
            ##
            fill!(A, one(FTYPE))
            fill!(psf, zero(FTYPE))
            for n=1:obs.nsubaps  # Loop through all subaps
                ## Aliases for subap-dependent parameters
                subap_image = obs.model_images[:, :, n, t]  # Model image for each subap at each time
                subap_mask = mask.masks[:, :, n, :]  # Mask for each subap at all wavelengths
                ##
                create_radiant_energy_pre_detector!(subap_image, image_small, image_big, psf, psf_temp, scale_psfs, x, patches.w, object_patch, obs.aperture_area, detector.exptime, subap_mask, A, P, p, refraction, iffts, convs, object.background / obs.dim^2 / obs.nsubaps, atmosphere.transmission, optics.response, ϕ_composite, ϕ_static, ϕ_slices, atmosphere.phase, smoothing, atmosphere.nlayers, extractor, atmosphere.sampling_nyquist_mperpix, atmosphere.heights, patches.npatches, reconstruction.nλ, reconstruction.nλint, reconstruction.Δλ)
                subap_image ./= detector.gain
                ω .= reconstruction.weight_function(obs.entropy[n, t], subap_image, detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
                if sum(subap_image) != sum(obs.images[:, :, n, t])
                    println("($dd, $t, $n) $(sum(subap_image)) $(sum(obs.images[:, :, n, t]))")
                end
                ϵ_local += loglikelihood_gaussian!(r, obs.images[:, :, n, t], subap_image, ω)  # Calculate the gaussian likelihood for the calculated model image and data frame
                # reconstruction.gradient_object(gradient_buffer, r, ω, image_big, psf, optics.response, atmosphere.transmission, detector.gain, detector.exptime, obs.aperture_area, obs.entropy[n, t], patches.w, patches.npatches, reconstruction.Δλ, reconstruction.nλ, corrs, container_builddim_real)
            end

            put!(helpers.channel_ift, iffts)  # Pre-allocated FFTs
            put!(helpers.channel_convolve, convs)  # Pre-allocated convolutions
            put!(helpers.channel_correlate, corrs)  # Pre-allocated correlations
            put!(helpers.channel_builddim_real, A)  # Buffer for amplitude
            put!(helpers.channel_builddim_real, ϕ_slices)  # Buffer for per-layer phase
            put!(helpers.channel_builddim_real, ϕ_composite)  # Buffer for composite phase
            put!(helpers.channel_smooth, smoothing)  # Function to smooth the composite phase
            put!(helpers.channel_builddim_cplx_4d, P)  # Pupil function buffers 
            put!(helpers.channel_builddim_cplx_4d, p)  # IFFT of pupil function buffers
            put!(helpers.channel_builddim_real_4d, psf)  # PSF buffer
            put!(helpers.channel_builddim_real, psf_temp)  # Temporary array needed to compute PSF
            put!(helpers.channel_builddim_real, object_patch)  # Object-times-patch weight buffer
            put!(helpers.channel_imagedim[dd], ω)
            put!(helpers.channel_imagedim[dd], r)
            put!(helpers.channel_builddim_real, image_big)  # Buffer to hold the full-size spectral image
            put!(helpers.channel_imagedim[dd], image_small)  # Buffer to hold the downsampled spectral image
            put!(helpers.channel_builddim_real, container_builddim_real)
            put!(helpers.channel_object_gradient_accumulator, copy(gradient_buffer))
            put!(helpers.channel_object_gradient_buffer, gradient_buffer)
            
            return ϵ_local
        end
    end

    for ~=1:Base.n_avail(helpers.channel_object_gradient_accumulator)
        buffer = take!(helpers.channel_object_gradient_accumulator)
        g .+= buffer
    end

    for w=1:reconstruction.nλ
        reconstruction.ϵ += regularizers.o_reg(x[:, :, w], g[:, :, w], regularizers.βo)
    end
    reconstruction.ϵ += regularizers.λ_reg(x, g, regularizers.βλ)

    return reconstruction.ϵ
end

@views function gradient_object_mle_gaussiannoise!(g, r, ω, image_big, psfs, optics_response, atmospheric_transmission, gain, exptime, area, entropy, patch_weights, npatches, Δλ, nλ, corr_prealloc!, container_builddim_real)
    r .*= ω
    block_replicate!(image_big, r)
    for np=1:npatches
        for w₁=1:nλ
            corr_prealloc!(container_builddim_real, image_big, psfs[:, :, np, w₁])
            g[:, :, w₁] .+= (2*Δλ*optics_response[w₁]*atmospheric_transmission[w₁]*exptime*area/gain) .* patch_weights[:, :, np] .* container_builddim_real
        end
    end
end

@views function gradient_object_mle_mixednoise!(g, r, ω, image_big, psfs, optics_response, atmospheric_transmission, gain, exptime, area, entropy, patch_weights, npatches, Δλ, nλ, corr_prealloc!, container_builddim_real)
    r .= 2 .* ω .* r .- (ω .* r).^2 .* entropy
    block_replicate!(image_big, r)
    for np=1:npatches
        for w₁=1:nλ
            corr_prealloc!(container_builddim_real, image_big, psfs[:, :, np, w₁])
            g[:, :, w₁] .+= (Δλ*optics_response[w₁]*atmospheric_transmission[w₁]*exptime*area/gain) .* patch_weights[:, :, np] .* container_builddim_real
        end
    end
end

@views function fg_opd_mle(x, g, observations, atmosphere, patches, reconstruction, object)
    FTYPE = gettype(reconstruction)  # Alias for the datatype
    ndatasets = length(observations)  # Number of datasets to be processed
    helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
    regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
    fill!(g, zero(FTYPE))  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values
    fill!(helpers.g_threads_wf, zero(FTYPE))  # Fill phase gradient buffer with zeros
    fill!(helpers.ϵ_threads, zero(FTYPE))  # Fill criterion vector with zeros
    if reconstruction.plot == true  # If plotting is enabled, object and phase plots will be updated here with the current proposed values
        update_phase_figure(x, atmosphere, reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        observation = observations[dd]  # Dataset from the full set
        detector = observation.detector
        optics = observation.optics  # Makes it easier to type
        mask = observations[dd].masks  # Masks for that dataset
        scale_psfs = mask.scale_psfs  # Scaler to multiply the PSFs by to ensure unit volume
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
        refraction_adj = helpers.refraction_adj[dd, :]  # Inverse refraction operator for the dataset
        r = helpers.r[dd]  # Buffer to hold residuals
        ω = helpers.ω[dd]  # Buffer to hold statistical weights
        ϕ_static = observation.phase_static  # Static phase for the dataset
        ## 
        fill!(observation.model_images, zero(FTYPE))  # Fill the model images with zeros to ensure a fresh start
        Threads.@threads :static for t=1:observation.nepochs  # Loop through all timesteps
            tid = Threads.threadid()  # Indexing is done by the thread id, which remains constant under the :static thread scheduler
            ## Aliases for time-dependent parameters
            iffts = helpers.ift[tid]  # Pre-allocated FFTs
            convs = helpers.convolve[tid]  # Pre-allocated convolutions
            extractor = helpers.extractor[dd][t, :, :, :]  # Interpolation operators for punch out
            extractor_adj = helpers.extractor_adj[dd][t, :, :, :]  # Interpolation operators for unpunch out
            A = helpers.A[:, :, tid]  # Buffer for amplitude
            ϕ_slices = helpers.ϕ_slices[:, :, tid]  # Buffer for per-layer phase
            ϕ_composite = helpers.ϕ_composite[:, :, tid]  # Buffer for composite phase
            smoothing = helpers.smooth[tid]  # Function to smooth the composite phase
            unsmoothing = helpers.unsmooth[tid]
            P = helpers.P[:, :, :, :, tid]  # Pupil function buffers 
            p = helpers.p[:, :, :, :, tid]  # IFFT of pupil function buffers
            psfs = helpers.psf[:, :, :, :, tid]  # PSF buffer
            psf_temp = helpers.psf_temp[:, :, tid]  # Temporary array needed to compute PSF
            object_patch = helpers.object_patch[:, :, tid]  # Object-times-patch weight buffer
            image_temp_big = helpers.image_temp_big[:, :, tid]  # Buffer to hold the full-size spectral image
            image_temp_small = helpers.image_temp_small[dd][:, :, tid]  # Buffer to hold the downsampled spectral image
            ##
            for n=1:observation.nsubaps  # Loop through all subaps
                ## Aliases for subap-dependent parameters
                subap_image = observation.model_images[:, :, n, t]  # Model image for each subap at each time
                subap_mask = mask.masks[:, :, n, :]  # Mask for each subap at all wavelengths
                ##
                create_image!(subap_image, image_temp_small, image_temp_big, psfs, psf_temp, scale_psfs, object.object, patches.w, object_patch, subap_mask, A, P, p, refraction, iffts, convs, atmosphere.transmission, optics.response, ϕ_composite, ϕ_static, ϕ_slices, x, smoothing, atmosphere.nlayers, extractor, atmosphere.sampling_nyquist_mperpix, atmosphere.heights, patches.npatches, reconstruction.nλ, reconstruction.nλint, reconstruction.Δλ)
                observation.model_images[:, :, n, t] .+= object.background_flux ./ observation.dim^2  # Add the background, which is specified per image, so scale by the number of pixels first
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])  # Calculate the gaussian likelihood for the calculated model image and data frame
                reconstruction.gradient_wf(helpers.g_threads_wf[:, :, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, unsmoothing, refraction_adj, extractor_adj, iffts, helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()  # Add together the gradients computed from each thread
        g .+= helpers.g_threads_wf[:, :, :, :, tid]
    end

    ## Apply regularization
    for l=1:atmosphere.nlayers
        for w₁=1:reconstruction.nλ
            for w₂=1:reconstruction.nλint 
                w = (w₁-1)*reconstruction.nλint + w₂
                ϵ += regularizers.wf_reg(x[:, :, l, w], g[:, :, l, w], regularizers.βwf)
            end
        end
    end

    reconstruction.ϵ = ϵ
    return ϵ
end

@views function gradient_opd_ffm_mle_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, o_corr, entropy, npatches, k_corr, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
    FTYPE = eltype(r)
    r .*= ω
    block_replicate!(c, r)
    conj!(p)
    for np=1:npatches
        for w₁=1:nλ
            d2 .= o_corr[w₁, np](c)  # <--
            for w₂=1:nλint
                w = (w₁-1)*nλint + w₂
                container_builddim_real .= d2
                mul!(d2, refraction_adj[w], container_builddim_real)

                p[:, :, np, w] .*= d2
                ifft!(d, p[:, :, np, w])
                d .*= P[:, :, np, w]
                d2 .= imag.(d)
                d2 .*= FTYPE(-8pi) * Δλ/λ[w] * response[w] * transmission[w]
                if smoothing == true
                    d2 .= k_corr(d2)
                end

                for l=1:nlayers
                    mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                    g[:, :, l] .+= container_sdim_real
                end
            end
        end
    end
end

@views function gradient_opd_ffm_mle_mixednoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, o_corr, entropy, npatches, k_corr, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
    FTYPE = eltype(r)
    r .= 2 .* ω .* r .- (ω .* r).^2 .* entropy
    block_replicate!(c, r)
    conj!(p)
    for np=1:npatches
        for w₁=1:nλ
            d2 .= o_corr[w₁, np](c)  # <--
            for w₂=1:nλint
                w = (w₁-1)*nλint + w₂
                container_builddim_real .= d2
                mul!(d2, refraction_adj[w], container_builddim_real)

                p[:, :, np, w] .*= d2
                ifft!(d, p[:, :, np, w])
                d .*= P[:, :, np, w]
                d2 .= imag.(d)
                d2 .*= FTYPE(-4pi) * Δλ/λ[w] * response[w] * transmission[w]

                if smoothing == true
                    d2 .= k_corr(d2)
                end

                for l=1:nlayers
                    mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                    g[:, :, l] .+= container_sdim_real
                end
            end
        end
    end
end

@views function fg_phase_ffm_mle(x, g, observations, atmosphere, patches, reconstruction, object)
    FTYPE = gettype(reconstruction)  # Alias for the datatype
    ndatasets = length(observations)  # Number of datasets to be processed
    helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
    regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
    fill!(g, zero(FTYPE))  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values 
    fill!(helpers.g_threads_wf, zero(FTYPE))  # Fill phase gradient buffer with zeros
    fill!(helpers.ϵ_threads, zero(FTYPE))  # Fill criterion vector with zeros
    if reconstruction.plot == true  # If plotting is enabled, object and phase plots will be updated here with the current proposed values
        update_phase_figure(x, atmosphere, reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        observation = observations[dd]  # Dataset from the full set
        detector = observation.detector
        optics = observation.optics  # Makes it easier to type
        mask = observations[dd].masks  # Masks for that dataset
        scale_psfs = mask.scale_psfs  # Scaler to multiply the PSFs by to ensure unit volume
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
        refraction_adj = helpers.refraction_adj[dd, :]  # Inverse refraction operator for the dataset
        r = helpers.r[dd]  # Buffer to hold residuals
        ω = helpers.ω[dd]  # Buffer to hold statistical weights
        ϕ_static = observation.phase_static  # Static phase for the dataset
        ## 
        fill!(observation.model_images, zero(FTYPE))  # Fill the model images with zeros to ensure a fresh start
        Threads.@threads :static for t=1:observation.nepochs  # Loop through all timesteps
            tid = Threads.threadid()  # Indexing is done by the thread id, which remains constant under the :static thread scheduler
            ## Aliases for time-dependent parameters
            iffts = helpers.ift[tid]  # Pre-allocated FFTs
            convs = helpers.convolve[tid]  # Pre-allocated convolutions
            extractor = helpers.extractor[dd][t, :, :, :]  # Interpolation operators for punch out
            extractor_adj = helpers.extractor_adj[dd][t, :, :, :]  # Interpolation operators for unpunch out
            A = helpers.A[:, :, tid]  # Buffer for amplitude
            ϕ_slices = helpers.ϕ_slices[:, :, tid]  # Buffer for per-layer phase
            ϕ_composite = helpers.ϕ_composite[:, :, tid]  # Buffer for composite phase
            smoothing = helpers.smooth[tid]  # Function to smooth the composite phase
            unsmoothing = helpers.unsmooth[tid]
            P = helpers.P[:, :, :, :, tid]  # Pupil function buffers 
            p = helpers.p[:, :, :, :, tid]  # IFFT of pupil function buffers
            psfs = helpers.psf[:, :, :, :, tid]  # PSF buffer
            psf_temp = helpers.psf_temp[:, :, tid]  # Temporary array needed to compute PSF
            object_patch = helpers.object_patch[:, :, tid]  # Object-times-patch weight buffer
            image_temp_big = helpers.image_temp_big[:, :, tid]  # Buffer to hold the full-size spectral image
            image_temp_small = helpers.image_temp_small[dd][:, :, tid]  # Buffer to hold the downsampled spectral image
            ##
            fill!(psfs, zero(FTYPE))
            for n=1:observation.nsubaps  # Loop through all subaps
                ## Aliases for subap-dependent parameters
                subap_model_image = observation.model_images[:, :, n, t]  # Model image for each subap at each time
                subap_mask = mask.masks[:, :, n, :]  # Mask for each subap at all wavelengths
                ##
                create_radiant_energy_pre_detector!(subap_model_image, image_temp_small, image_temp_big, psfs, psf_temp, scale_psfs, object.object, patches.w, object_patch, observation.aperture_area, detector.exptime, subap_mask, A, P, p, refraction, iffts, convs, object.background / (observation.dim^2 * observation.nsubaps), atmosphere.transmission, optics.response, ϕ_composite, ϕ_static, ϕ_slices, x, smoothing, atmosphere.nlayers, extractor, atmosphere.sampling_nyquist_mperpix, atmosphere.heights, patches.npatches, reconstruction.nλ, reconstruction.nλint, reconstruction.Δλ)
                subap_model_image ./= detector.gain  # Add the background, which is specified per image, so scale by the number of pixels first
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], subap_model_image, detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], subap_model_image, ω[:, :, tid])  # Calculate the gaussian likelihood for the calculated model image and data frame
                reconstruction.gradient_wf(helpers.g_threads_wf[:, :, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, detector.gain, detector.exptime, observation.aperture_area, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, unsmoothing, refraction_adj, extractor_adj, iffts, helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()  # Add together the gradients computed from each thread
        g .+= helpers.g_threads_wf[:, :, :, :, tid]
    end

    ## Apply regularization
    for l=1:atmosphere.nlayers
        for w₁=1:reconstruction.nλ
            for w₂=1:reconstruction.nλint 
                w = (w₁-1)*reconstruction.nλint + w₂
                ϵ += regularizers.wf_reg(x[:, :, l, w], g[:, :, l, w], regularizers.βwf)
            end
        end
    end

    reconstruction.ϵ = ϵ
    return ϵ
end

@views function gradient_phase_ffm_mle_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, nlayers, precorr_object, entropy, npatches, unsmooth!, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
    r .*= ω
    block_replicate!(c, r)
    conj!(p)
    for np=1:npatches
        for w₁=1:nλ
            precorr_object[w₁, np](d2, c)
            for w₂=1:nλint
                w = (w₁-1)*nλint + w₂
                container_builddim_real .= d2
                mul!(d2, refraction_adj[w], container_builddim_real)

                p[:, :, np, w] .*= d2
                ifft!(d, p[:, :, np, w])
                d .*= P[:, :, np, w]
                d2 .= imag.(d)
                d2 .*= -4 * response[w] * transmission[w] * gain * exptime * area
                unsmooth!(d2, d2)

                for l=1:nlayers
                    mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                    g[:, :, l, w] .+= container_sdim_real
                end
            end
        end
    end
end

@views function gradient_phase_ffm_mle_mixednoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, nlayers, precorr_object, entropy, npatches, unsmooth!, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
    r .= 2 .* ω .* r .- (ω .* r).^2 .* entropy
    block_replicate!(c, r)
    conj!(p)
    for np=1:npatches
        for w₁=1:nλ
            precorr_object[w₁, np](d2, c)
            for w₂=1:nλint
                w = (w₁-1)*nλint + w₂
                container_builddim_real .= d2
                mul!(d2, refraction_adj[w], container_builddim_real)

                p[:, :, np, w] .*= d2
                ifft!(d, p[:, :, np, w])
                d .*= P[:, :, np, w]
                d2 .= imag.(d)
                d2 .*= -2 * response[w] * transmission[w] * gain * exptime * area
                unsmooth!(d2, d2)

                for l=1:nlayers
                    mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                    g[:, :, l, w] .+= container_sdim_real
                end
            end
        end
    end
end

# @views function fg_phase_mle(x, g, observations, atmosphere, masks, patches, reconstruction, object)
#     FTYPE = gettype(reconstruction)  # Alias for the datatype
#     ndatasets = length(observations)  # Number of datasets to be processed
#     helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
#     regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
#     fill!(g, zero(FTYPE))  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values
#     fill!(helpers.g_threads_wf, zero(FTYPE))  # Fill phase gradient buffer with zeros
#     fill!(helpers.ϵ_threads, zero(FTYPE))  # Fill criterion vector with zeros

#     for dd=1:ndatasets  # Loop through data channels
#         ## Aliases for dataset-dependent parameters
#         observation = observations[dd]  # Dataset from the full set
#         detector = observation.detector
#         optics = observation.optics  # Makes it easier to type
#         mask = masks[dd]  # Masks for that dataset
#         scale_psfs = mask.scale_psfs  # Scaler to multiply the PSFs by to ensure unit volume
#         refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
#         refraction_adj = helpers.refraction_adj[dd, :]  # Inverse refraction operator for the dataset
#         r = helpers.r[dd]  # Buffer to hold residuals
#         ω = helpers.ω[dd]  # Buffer to hold statistical weights
#         ϕ_static = observation.phase_static  # Static phase for the dataset
#         ## 
#         fill!(observation.model_images, zero(FTYPE))  # Fill the model images with zeros to ensure a fresh start
#         Threads.@threads :static for t=1:observation.nepochs  # Loop through all timesteps
#             tid = Threads.threadid()  # Indexing is done by the thread id, which remains constant under the :static thread scheduler
#             tix = (dd==1) ? t : observations[dd-1].nepochs + t
#             ϕ = x[:, :, tix, :]
#             ## Aliases for time-dependent parameters
#             iffts = helpers.ift[tid]  # Pre-allocated FFTs
#             convs = helpers.convolve[tid]  # Pre-allocated convolutions
#             A = atmosphere.A[:, :, tix, :]  # Buffer for amplitude
#             smoothing = helpers.smooth[tid]  # Function to smooth the composite phase
#             unsmoothing = helpers.unsmooth[tid]
#             P = helpers.P[:, :, 1, :, tid]  # Pupil function buffers 
#             p = helpers.p[:, :, 1, :, tid]  # IFFT of pupil function buffers
#             psfs = helpers.psf[:, :, :, :, tid]  # PSF buffer
#             psf_temp = helpers.psf_temp[:, :, tid]  # Temporary array needed to compute PSF
#             image_temp_big = helpers.image_temp_big[:, :, tid]  # Buffer to hold the full-size spectral image
#             image_temp_small = helpers.image_temp_small[dd][:, :, tid]  # Buffer to hold the downsampled spectral image
#             ##
#             fill!(psfs, zero(FTYPE))
#             for n=1:observation.nsubaps  # Loop through all subaps
#                 ## Aliases for subap-dependent parameters
#                 subap_model_image = observation.model_images[:, :, n, t]  # Model image for each subap at each time
#                 subap_mask = mask.masks[:, :, n, :]  # Mask for each subap at all wavelengths
#                 ##
#                 create_radiant_energy_pre_detector!(subap_model_image, image_temp_small, image_temp_big, psfs, psf_temp, scale_psfs, object.object, observation.aperture_area, detector.exptime, subap_mask, A, P, p, refraction, iffts, convs, object.background / (observation.dim^2 * observation.nsubaps), atmosphere.transmission, optics.response, ϕ, ϕ_static, smoothing, reconstruction.nλ, reconstruction.nλint, reconstruction.Δλ)
#                 subap_model_image ./= detector.gain  # Add the background, which is specified per image, so scale by the number of pixels first
#                 ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], subap_model_image, detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
#                 helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], subap_model_image, ω[:, :, tid])  # Calculate the gaussian likelihood for the calculated model image and data frame
#                 reconstruction.gradient_wf(helpers.g_threads_wf[:, :, tix, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, detector.gain, detector.exptime, observation.aperture_area, helpers.o_corr[:, :, tid], observation.entropy[n, t], unsmoothing, refraction_adj, iffts, helpers.containers_builddim_real[:, :, tid])
#             end
#         end
#     end

#     ϵ = sum(helpers.ϵ_threads)
#     for tid=1:Threads.nthreads()  # Add together the gradients computed from each thread
#         g .+= helpers.g_threads_wf[:, :, :, :, tid]
#     end

#     ## Apply regularization
#     for l=1:atmosphere.nlayers
#         for w₁=1:reconstruction.nλ
#             for w₂=1:reconstruction.nλint 
#                 w = (w₁-1)*reconstruction.nλint + w₂
#                 ϵ += regularizers.wf_reg(x[:, :, l, w], g[:, :, l, w], regularizers.βwf)
#             end
#         end
#     end

#     reconstruction.ϵ = ϵ
#     return ϵ
# end

# @views function gradient_phase_mle_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, precorr_object, entropy, unsmooth!, refraction_adj, ifft!, container_builddim_real)

# end

# @views function gradient_phase_mle_mixednoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, gain, exptime, area, precorr_object, entropy, unsmooth!, refraction_adj, ifft!, container_builddim_real)

# end
