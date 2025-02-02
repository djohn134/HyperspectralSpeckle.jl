function loglikelihood_gaussian(data, model, ω)
    FTYPE = eltype(data)
    r = zeros(FTYPE, size(data))
    ϵ = FTYPE(loglikelihood_gaussian(r, data, model, ω))
    return ϵ
end

function loglikelihood_gaussian(r, data, model, ω)
    FTYPE = eltype(r)
    r .= model .- data
    ϵ = FTYPE(mapreduce(x -> x[1] * x[2]^2, +, zip(ω, r)))
    return ϵ
end

@views function fg_object_mle(x::AbstractArray{<:AbstractFloat, 3}, g, observations, atmosphere, masks, patches, reconstruction, object)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    regularizers = reconstruction.regularizers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_obj, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))
    if reconstruction.plot == true
        update_object_figure(dropdims(mean(x, dims=3), dims=3), reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        observation = observations[dd]  # Dataset from the full set
        optics = observation.optics  # Makes it easier to type
        mask = masks[dd]  # Masks for that dataset
        scale_psfs = mask.scale_psfs  # Scaler to multiply the PSFs by to ensure unit volume
        refraction = helpers.refraction[dd, :]  # Refraction operator for the dataset
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
            corrs = helpers.correlate[tid]  # Pre-allocated correlations
            extractor = helpers.extractor[dd, t, :, :, :]  # Interpolation operators for punch out
            A = helpers.A[:, :, tid]  # Buffer for amplitude
            ϕ_slices = helpers.ϕ_slices[:, :, tid]  # Buffer for per-layer phase
            ϕ_composite = helpers.ϕ_composite[:, :, tid]  # Buffer for composite phase
            smoothing = helpers.smooth[tid]  # Function to smooth the composite phase
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
                create_image!(subap_image, image_temp_small, image_temp_big, psfs, psf_temp, scale_psfs, x, patches.w, object_patch, subap_mask, A, P, p, refraction, iffts, convs, atmosphere.transmission, optics.response, ϕ_composite, ϕ_static, ϕ_slices, atmosphere.phase, smoothing, atmosphere.nlayers, extractor, atmosphere.sampling_nyquist_mperpix, atmosphere.heights, patches.npatches, reconstruction.nλ, reconstruction.nλint, reconstruction.Δλ)
                observation.model_images[:, :, n, t] .+= object.background_flux ./ observation.dim^2  # Add the background, which is specified per image, so scale by the number of pixels first
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])  # Calculate the gaussian likelihood for the calculated model image and data frame
                reconstruction.gradient_object(helpers.g_threads_obj[:, :, :, tid], r[:, :, tid], ω[:, :, tid], image_temp_big, psfs, observation.entropy[n, t], patches.w, patches.npatches, reconstruction.Δλ, reconstruction.nλ, corrs, helpers.containers_builddim_real[:, :, tid])
            end
        end
    end

    ϵ = FTYPE(sum(helpers.ϵ_threads))
    for tid=1:Threads.nthreads()
        g .+= helpers.g_threads_obj[:, :, :, tid]
    end

    for w=1:reconstruction.nλ
        ϵ += regularizers.o_reg(x[:, :, w], g[:, :, w], regularizers.βo)
    end
    ϵ += regularizers.λ_reg(x, g, regularizers.βλ)

    reconstruction.ϵ = ϵ
    return ϵ
end

@views function gradient_object_mle_gaussiannoise!(g::AbstractArray{<:AbstractFloat, 3}, r, ω, image_big, psfs, entropy, patch_weights, npatches, Δλ, nλ, corr_prealloc!, container_builddim_real)
    r .*= ω
    block_replicate!(image_big, r)
    for np=1:npatches
        for w₁=1:nλ
            corr_prealloc!(container_builddim_real, image_big, psfs[:, :, np, w₁])
            g[:, :, w₁] .+= (2*Δλ) .* patch_weights[:, :, np] .* container_builddim_real
        end
    end
end

@views function gradient_object_mle_mixednoise!(g::AbstractArray{<:AbstractFloat, 3}, r, ω, image_big, psfs, entropy, patch_weights, npatches, Δλ, nλ, corr_prealloc!, container_builddim_real)
    r .= 2 .* ω .* r .- (ω .* r).^2 .* entropy
    block_replicate!(image_big, r)
    for np=1:npatches
        for w₁=1:nλ
            corr_prealloc!(container_builddim_real, image_big, psfs[:, :, np, w₁])
            g[:, :, w₁] .+= Δλ .* patch_weights[:, :, np] .* container_builddim_real
        end
    end
end

@views function fg_opd_mle(x, g, observations, atmosphere, masks, patches, reconstruction, object)
    FTYPE = gettype(reconstruction)  # Alias for the datatype
    ndatasets = length(observations)  # Number of datasets to be processed
    helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
    regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
    fill!(g, zero(FTYPE))  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values
    fill!(helpers.g_threads_ϕ, zero(FTYPE))  # Fill phase gradient buffer with zeros
    fill!(helpers.ϵ_threads, zero(FTYPE))  # Fill criterion vector with zeros
    if reconstruction.plot == true  # If plotting is enabled, object and phase plots will be updated here with the current proposed values
        update_phase_figure(x, atmosphere, reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        observation = observations[dd]  # Dataset from the full set
        optics = observation.optics  # Makes it easier to type
        mask = masks[dd]  # Masks for that dataset
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
            extractor = helpers.extractor[dd, t, :, :, :]  # Interpolation operators for punch out
            extractor_adj = helpers.extractor_adj[dd, t, :, :, :]  # Interpolation operators for unpunch out
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
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])  # Calculate the gaussian likelihood for the calculated model image and data frame
                reconstruction.gradient_wf(helpers.g_threads_opd[:, :, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, unsmoothing, refraction_adj, extractor_adj, iffts, helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()  # Add together the gradients computed from each thread
        g .+= helpers.g_threads_ϕ[:, :, :, :, tid]
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

@views function gradient_opd_mle_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, o_corr, entropy, npatches, k_corr, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
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

@views function gradient_opd_mle_mixednoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, o_corr, entropy, npatches, k_corr, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
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

@views function fg_phase_mle(x, g, observations, atmosphere, masks, patches, reconstruction, object)
    FTYPE = gettype(reconstruction)  # Alias for the datatype
    ndatasets = length(observations)  # Number of datasets to be processed
    helpers = reconstruction.helpers  # Alias for helpers, makes it quicker to type
    regularizers = reconstruction.regularizers  # Alias for regularizers, makes it quicker to type
    fill!(g, zero(FTYPE))  # Fill gradient with zeros, OptimPack initializes it with undef's, which can give crazy NaN values
    fill!(helpers.g_threads_ϕ, zero(FTYPE))  # Fill phase gradient buffer with zeros
    fill!(helpers.ϵ_threads, zero(FTYPE))  # Fill criterion vector with zeros
    if reconstruction.plot == true  # If plotting is enabled, object and phase plots will be updated here with the current proposed values
        update_phase_figure(x, atmosphere, reconstruction)
    end
    
    for dd=1:ndatasets  # Loop through data channels
        ## Aliases for dataset-dependent parameters
        observation = observations[dd]  # Dataset from the full set
        optics = observation.optics  # Makes it easier to type
        mask = masks[dd]  # Masks for that dataset
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
            extractor = helpers.extractor[dd, t, :, :, :]  # Interpolation operators for punch out
            extractor_adj = helpers.extractor_adj[dd, t, :, :, :]  # Interpolation operators for unpunch out
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
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)  # The statistical weight is given as either 1/σ^2 for purely gaussian noise, or 1/√(Î+σ^2) for gaussian and Poisson noise
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])  # Calculate the gaussian likelihood for the calculated model image and data frame
                reconstruction.gradient_wf(helpers.g_threads_ϕ[:, :, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, unsmoothing, refraction_adj, extractor_adj, iffts, helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()  # Add together the gradients computed from each thread
        g .+= helpers.g_threads_ϕ[:, :, :, :, tid]
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

@views function gradient_phase_mle_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, precorr_object, entropy, npatches, unsmooth!, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
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
                d2 .*= -4 * response[w] * transmission[w]
                unsmooth!(d2, d2)

                for l=1:nlayers
                    mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                    g[:, :, l, w] .+= container_sdim_real
                end
            end
        end
    end
end

@views function gradient_phase_mle_mixednoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, precorr_object, entropy, npatches, unsmooth!, refraction_adj, extractor_adj, ifft!, container_builddim_real, container_sdim_real)
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
                d2 .*= -2 * response[w] * transmission[w]
                unsmooth!(d2, d2)

                for l=1:nlayers
                    mul!(container_sdim_real, extractor_adj[np, l, w], d2)
                    g[:, :, l, w] .+= container_sdim_real
                end
            end
        end
    end
end

@views function fg_static_phase_mle(x, g, observations, atmosphere, masks, patches, reconstruction, object)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    patch_helpers = reconstruction.patch_helpers
    regularizers = reconstruction.regularizers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_ϕ, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))
    if reconstruction.plot == true
        update_static_phase_figure(x, atmosphere, reconstruction)
    end
    
    for dd=1:ndatasets
        observation = observations[dd]
        optics = observation.optics
        psfs = patches.psfs[dd]
        mask = masks[dd]
        scale_psf = mask.scale_psfs
        refraction = helpers.refraction[dd, :]
        refraction_adj = helpers.refraction_adj[dd, :]
        r = helpers.r[dd]
        ω = helpers.ω[dd]
        Î_small = helpers.Î_small[dd]
        ϕ_static = observation.phase_static
        fill!(observation.model_images, zero(FTYPE))
        fill!(psfs, zero(FTYPE))
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            extractor = patch_helpers.extractor[t, :, :, :]
            extractor_adj = patch_helpers.extractor_adj[t, :, :, :]
            Î_big = helpers.Î_big[:, :, tid]
            A = patches.A[dd][:, :, :, :, t, :]
            ϕ_slices = patches.phase_slices[:, :, :, t, :, :]
            ϕ_composite = patches.phase_composite[:, :, :, t, :]
            for n=1:observation.nsubaps
                P = patch_helpers.P[:, :, :, :, tid]
                p = patch_helpers.p[:, :, :, :, tid]
                for np=1:patches.npatches
                    for w₁=1:reconstruction.nλ
                        for w₂=1:reconstruction.nλint 
                            w = (w₁-1)*reconstruction.nλint + w₂
                            fill!(ϕ_composite[:, :, np, w], zero(FTYPE))
                            for l=1:atmosphere.nlayers
                                ## Aliases don't allocate
                                fill!(ϕ_slices[:, :, np, l, w], zero(FTYPE))
                                position2phase!(ϕ_slices[:, :, np, l, w], x[:, :, l, w], extractor[np, l, w])
                                ϕ_composite[:, :, np, w] .+= ϕ_slices[:, :, np, l, w]
                            end
                            ϕ_composite[:, :, np, w] .+= ϕ_static[:, :, w]
                            
                            if reconstruction.smoothing == true
                                ϕ_composite[:, :, np, w] .= helpers.k_conv[tid](ϕ_composite[:, :, np, w])
                            end

                            pupil2psf!(Î_big, helpers.containers_builddim_real[:, :, tid], mask.masks[:, :, n, w], P[:, :, np, w], p[:, :, np, w], A[:, :, np, n, w], ϕ_composite[:, :, np, w], optics.response[w], atmosphere.transmission[w], scale_psf[w], helpers.ift[1][tid], refraction[w])
                            psfs[:, :, np, n, t, w₁] .+= Î_big ./ reconstruction.nλint
                        end
                    end
                    create_polychromatic_image!(observation.model_images[:, :, n, t], Î_small[:, :, tid], Î_big, helpers.o_conv[:, np, tid], psfs[:, :, np, n, t, :], reconstruction.λ, reconstruction.Δλ)
                end
                observation.model_images[:, :, n, t] .+= object.background_flux ./ observation.dim^2
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)
                helpers.ϵ_threads[tid] += loglikelihood_gaussian(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid])
                reconstruction.gradient_wf(helpers.g_threads_ϕ[:, :, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, helpers.k_corr[tid], refraction_adj, extractor_adj, helpers.ift[1][tid], helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()
        g .+= helpers.g_threads_ϕ[:, :, :, :, tid]
    end

    # Apply regularization
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
