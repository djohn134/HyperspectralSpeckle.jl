function mrl(data, model, ω, mask_acf)
    FTYPE = eltype(r)
    r = zeros(FTYPE, size(data))
    autocorr = setup_autocorr(size(data, 1), FTYPE=FTYPE)
    container = zeros(FTYPE, size(data))
    ϵ = FTYPE(mrl(r, data, model, ω, mask_acf, autocorr, container))
    return ϵ
end

function mrl(r, data, model, ω, mask_acf, autocorr, container)
    FTYPE = eltype(r)
    r .= model .- data
    autocorr(container, r)
    container .*= ω
    ϵ = FTYPE(mapreduce(x -> x[1] * x[2]^2, +, zip(mask_acf, container)))
    return ϵ
end

@views function fg_object_mrl(x, g, observations, reconstruction, patches)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    regularizers = reconstruction.regularizers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_obj, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))
    update_object_figure(dropdims(mean(x, dims=3), dims=3), reconstruction)

    for dd=1:ndatasets
        mask_acf = helpers.mask_acf[dd]
        corr! = helpers.correlate[dd+1]
        autocorr! = helpers.autocorr[dd+1]
        observation = observations[dd]
        psfs = patches.psfs[dd]
        r = helpers.r[dd]
        ω = helpers.ω[dd]
        Î_small = helpers.Î_small[dd]
        container_pdim_cplx = helpers.containers_pdim_cplx[dd]
        container_pdim_real = helpers.containers_pdim_real[dd]
        fill!(observation.model_images, zero(FTYPE))
        Threads.@threads :static for t=1:observation.nepochs
            tid = Threads.threadid()
            Î_big = helpers.Î_big[:, :, tid]
            for n=1:observation.nsubaps
                for np=1:patches.npatches
                    create_polychromatic_image!(observation.model_images[:, :, n, t], Î_small[:, :, tid], Î_big, patches.w[:, :, np], helpers.containers_builddim_real[:, :, tid], x, psfs[:, :, np, n, t, :], reconstruction.λ, reconstruction.Δλ)
                end
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)                
                helpers.ϵ_threads[tid] += mrl(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid], mask_acf, autocorr![tid], container_pdim_real[:, :, tid])
                reconstruction.gradient_object(helpers.g_threads_obj[:, :, :, tid], r[:, :, tid], ω[:, :, tid], Î_big, psfs[:, :, :, n, t, :], patches.w, patches.npatches, reconstruction.Δλ, reconstruction.nλ, mask_acf, container_pdim_real[:, :, tid], container_pdim_cplx[:, :, tid], helpers.containers_builddim_cplx[:, :, tid], autocorr![tid], corr![tid], helpers.correlate[1][tid])
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

@views function gradient_object_mrl_gaussiannoise!(g, r, ω, image_big, psfs, patch_weights, npatches, Δλ, nλ, mask_acf, container_pdim_real, container_pdim_cplx, container_builddim_cplx, autocorr!, corr_pdim!, corr_builddim!)
    autocorr!(container_pdim_real, r)
    corr_pdim!(container_pdim_cplx, r, container_pdim_real)
    container_pdim_real .= ω .* mask_acf .* real.(container_pdim_cplx)
    block_replicate!(image_big, container_pdim_real)
    for np=1:npatches
        for w=1:nλ
            corr_builddim!(container_builddim_cplx, image_big, psfs[:, :, np, w])
            # g[:, :, w] .+= Δλ .* patch_weights[:, :, np] .* ccorr_psf(image_big, psfs[:, :, np, w])
            g[:, :, w] .+= (4 * Δλ) .* patch_weights[:, :, np] .* real.(container_builddim_cplx)
        end
    end
end

@views function fg_opd_mrl(x, g, observations, atmosphere, masks, patches, reconstruction)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    patch_helpers = reconstruction.patch_helpers
    regularizers = reconstruction.regularizers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_opd, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))
    update_opd_figure(x, atmosphere, reconstruction)

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
        mask_acf = helpers.mask_acf[dd]
        ift! = helpers.ift[dd+1]
        corr! = helpers.correlate[dd+1]
        autocorr! = helpers.autocorr[dd+1]
        container_pdim_cplx = helpers.containers_pdim_cplx[dd]
        container_pdim_real = helpers.containers_pdim_real[dd]
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
                                helpers.containers_sdim_real[:, :, tid] .= FTYPE(2pi) ./ reconstruction.λtotal[w] .* x[:, :, l]
                                position2phase!(ϕ_slices[:, :, np, l, w], helpers.containers_sdim_real[:, :, tid], extractor[np, l, w])
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
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)
                helpers.ϵ_threads[tid] += mrl(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid], mask_acf, autocorr![tid], container_pdim_real[:, :, tid])
                reconstruction.gradient_wf(helpers.g_threads_opd[:, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, reconstruction.smoothing, helpers.k_corr[tid], refraction_adj, extractor_adj, mask_acf, helpers.ift[1][tid], autocorr![tid], corr![tid], container_pdim_real[:, :, tid], container_pdim_cplx[:, :, tid], helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = sum(helpers.ϵ_threads)
    for tid=1:Threads.nthreads()
        g .+= helpers.g_threads_opd[:, :, :, tid]
    end

    # Apply regularization
    for l=1:atmosphere.nlayers
        ϵ += regularizers.wf_reg(x[:, :, l], g[:, :, l], regularizers.βwf)
    end

    reconstruction.ϵ = ϵ
    return ϵ
end

@views function gradient_opd_mrl_gaussiannoise!(g, r, ω, P, p, c, d, d2, λ, Δλ, nλ, nλint, response, transmission, nlayers, o_corr, entropy, npatches, smoothing, k_corr, refraction_adj, extractor_adj, mask_acf, ifft!, autocorr!, corr!, container_pdim_real, container_pdim_cplx, container_builddim_real, container_sdim_real)
    FTYPE = eltype(r)
    autocorr!(container_pdim_real, r)
    corr!(container_pdim_cplx, r, container_pdim_real)
    container_pdim_real .= ω .* mask_acf .* real.(container_pdim_cplx)
    block_replicate!(c, container_pdim_real)
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
                d2 .*= FTYPE(-16pi) * Δλ/λ[w] * response[w] * transmission[w]
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

@views function fg_phase_mrl(x, g, observations, atmosphere, masks, patches, reconstruction)
    ## Optimized but unreadable
    FTYPE = gettype(reconstruction)
    ndatasets = length(observations)
    helpers = reconstruction.helpers
    patch_helpers = reconstruction.patch_helpers
    regularizers = reconstruction.regularizers
    fill!(g, zero(FTYPE))
    fill!(helpers.g_threads_opd, zero(FTYPE))
    fill!(helpers.ϵ_threads, zero(FTYPE))
    update_phase_figure(x, atmosphere, reconstruction)

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
        mask_acf = helpers.mask_acf[dd]
        ift! = helpers.ift[dd+1]
        corr! = helpers.correlate[dd+1]
        autocorr! = helpers.autocorr[dd+1]
        container_pdim_cplx = helpers.containers_pdim_cplx[dd]
        container_pdim_real = helpers.containers_pdim_real[dd]
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
                                helpers.containers_sdim_real[:, :, tid] .= x[:, :, l]
                                position2phase!(ϕ_slices[:, :, np, l, w], helpers.containers_sdim_real[:, :, tid], extractor[np, l, w])
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
                ω[:, :, tid] .= reconstruction.weight_function(observation.entropy[n, t], observation.model_images[:, :, n, t], observation.detector.rn)
                helpers.ϵ_threads[tid] += mrl(r[:, :, tid], observation.images[:, :, n, t], observation.model_images[:, :, n, t], ω[:, :, tid], mask_acf, autocorr![tid], container_pdim_real[:, :, tid])
                reconstruction.gradient_wf(helpers.g_threads_ϕ[:, :, :, :, tid], r[:, :, tid], ω[:, :, tid], P, p, helpers.c[:, :, tid], helpers.d[:, :, tid], helpers.d2[:, :, tid], reconstruction.λtotal, reconstruction.Δλtotal, reconstruction.nλ, reconstruction.nλint, optics.response, atmosphere.transmission, atmosphere.nlayers, helpers.o_corr[:, :, tid], observation.entropy[n, t], patches.npatches, reconstruction.smoothing, helpers.k_corr[tid], refraction_adj, extractor_adj, mask_acf, helpers.ift[1][tid], autocorr![tid], corr![tid], container_pdim_real[:, :, tid], container_pdim_cplx[:, :, tid], helpers.containers_builddim_real[:, :, tid], helpers.containers_sdim_real[:, :, tid])
            end
        end
    end

    ϵ = FTYPE(sum(helpers.ϵ_threads))
    for tid=1:Threads.nthreads()
        g .+= helpers.g_threads_obj[:, :, 1, tid]
    end

    ϵ += regularizers.o_reg(x, g, regularizers.βo)
    ϵ += regularizers.λ_reg(x, g, regularizers.βλ)
    # o_obs[] = rotr90(g[:, :, end])

    return ϵ
end

@views function gradient_object_mrl!(g::AbstractMatrix{<:AbstractFloat}, r, ω, image_big, psfs, patch_weights, npatches, Δλ, nλ, M, container_pdim_cplx, container_pdim_real, ifft_prealloc, autocorr)
    autocorr(container_pdim_real, r)
    ifft_prealloc(container_pdim_cplx, r)
    conj!(container_pdim_cplx)
    container_pdim_cplx .*= container_pdim_real
    ifft_prealloc(container_pdim_cplx, container_pdim_cplx)
    container_pdim_real .= 4 .* M .* ω .* real.(container_pdim_cplx)
    block_replicate!(image_big, container_pdim_real)
    for np=1:npatches
        for w=1:nλ
            g .+= Δλ .* patch_weights[:, :, np] .* ccorr_psf(image_big, psfs[:, :, np, w])
        end
    end
end
