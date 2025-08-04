function Base.display(patches::AnisoplanaticPatches{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Anisoplanatic Patches\n"); print(Crayon(reset=true))
    println("\tSize: $(patches.dim)×$(patches.dim) pixels")
    println("\tOverlap: $(patches.overlap)")
    println("\tNumber of patches: $(Int(sqrt(patches.npatches)))×$(Int(sqrt(patches.npatches))) patches")
end

function Base.display(atmosphere::Atmosphere{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Atmosphere\n"); print(Crayon(reset=true))
    println("\tNumber of layers: $(atmosphere.nlayers) layers")
    println("\tWind speed: $(atmosphere.wind[:, 1]) m/s")
    println("\tWind direction: $(atmosphere.wind[:, 2]) deg")
    println("\tInner scale: $(atmosphere.l0) m")
    println("\tOuter scale: $(atmosphere.L0) m")
    println("\tLayer Heights: $(atmosphere.heights) m")
    println("\tFried paremeter: $(atmosphere.r0) m")
    println("\tReference wavelength: $(atmosphere.λ_ref) nm")
    println("\tWavelength: $(minimum(atmosphere.λ)) — $(maximum(atmosphere.λ)) m")
    println("\tNumber of wavelengths: $(length(atmosphere.λ)) wavelengths")
    println("\tPropagate: $(atmosphere.propagate)")
end

function Base.display(masks::Masks{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Masks\n"); print(Crayon(reset=true))
    println("\tSize: $(masks.dim)×$(masks.dim) pixels")
    println("\tConfiguration: $(masks.nsubaps_side)×$(masks.nsubaps_side) subapertures")
    println("\tWavelength: $(minimum(masks.λ)) — $(maximum(masks.λ)) m")
    println("\tNumber of wavelengths: $(length(masks.λ)) wavelengths")
end

function Base.display(object::Object{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Object\n"); print(Crayon(reset=true))
    println("\tSize: $(object.dim)×$(object.dim) pixels")
    println("\tFOV: $(object.fov)×$(object.fov) arcsec")
    println("\tRange: $(object.range) m")
    println("\tIrradiance: $(object.irradiance) ph/s/m^2")
    println("\tBackground Irradiance: $(object.background) ph/s/m^2")
    println("\tWavelength: $(minimum(object.λ)) — $(maximum(object.λ)) m")
    println("\tNumber of wavelengths: $(length(object.λ))")
end

function Base.display(detector::Detector{<:AbstractFloat})
    DTYPE = gettypes(detector)[end]
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Detector\n"); print(Crayon(reset=true))
    println("\tBit Depth: $(DTYPE)")
    println("\tRN: $(detector.rn) e⁻")
    println("\tGain: $(detector.gain) e⁻/ADU")
    println("\tSaturation: $(detector.saturation) e⁻")
    println("\tExposure time: $(detector.exptime) s")
    println("\tPlate Scale: $(detector.pixscale) arcsec/pix")
    println("\tWavelength: $(minimum(detector.λ)) — $(maximum(detector.λ)) m")
    println("\tNyquist sampled wavelength: $(detector.λ_nyquist) m")
end

function Base.display(optical_system::OpticalSystem{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Optical system\n"); print(Crayon(reset=true))
    println("\tNumber of elements: $(length(optical_system.elements))")
    println("\tWavelength: $(minimum(optical_system.λ)) — $(maximum(optical_system.λ)) m")
end

function Base.display(observations::Observations{<:AbstractFloat, <:Real})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Observations\n"); print(Crayon(reset=true))
    println("\tImage Size: $(observations.dim)×$(observations.dim) pixels")
    println("\tNumber of frames: $(observations.nepochs)")
    println("\tNumber of subapertures: $(observations.nsubaps_side)×$(observations.nsubaps_side) subapertures")
    println("\tTelescope Diameter: $(observations.D) m")
    println("\tZenith angle: $(observations.ζ) deg")
end

function Base.display(reconstruction::Reconstruction{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Reconstruction\n"); print(Crayon(reset=true))
    println("\tImage build size: $(reconstruction.build_dim)×$(reconstruction.build_dim) pixels")
    println("\tWavelength: $(minimum(reconstruction.λ)) — $(maximum(reconstruction.λ)) m")
    println("\tNumber of wavelength: $(reconstruction.nλ)")
    println("\tNumber of integrated wavelengths: $(reconstruction.nλint)")
    println("\tNumber of data channels: $(reconstruction.ndatasets)")
    println("\tWavefront Parameter: $(symbol2str[reconstruction.wavefront_parameter])")
    println("\tNoise weighting: $(reconstruction.weight_function)")
    println("\tObject gradient function: $(reconstruction.gradient_object)")
    println("\tWavefront gradient function: $(reconstruction.gradient_wf)")
    println("\tNumber of MFBD cycles: $(reconstruction.niter_mfbd)")
    println("\tMax iterations: $(reconstruction.maxiter)") 
    println("\tMax evaluations: $(reconstruction.maxeval["wf"]) (wf), $(reconstruction.maxeval["object"]) (object)")
    println("\tSmoothing: $(reconstruction.smoothing) (schedule: $(reconstruction.fwhm_schedule), Max FWHM: $(reconstruction.maxFWHM), Min FWHM: $(reconstruction.minFWHM))")
    println("\tStopping criteria: $(reconstruction.grtol) (grtol), $(reconstruction.frtol) (frtol), $(reconstruction.xrtol) (xrtol)")
end

function Base.display(regularizers::Regularizers{<:AbstractFloat})
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Regularizers\n"); print(Crayon(reset=true))
    println("\tObject spatial regularizer: $(regularizers.o_reg), β₀=$(regularizers.βo) (schedule: $(regularizers.βo_schedule))")
    println("\tObject wavelength regularizer: $(regularizers.λ_reg), β₀=$(regularizers.βλ) (schedule: $(regularizers.βλ_schedule))")
    println("\tWavefront spatial regularizer: $(regularizers.wf_reg), β₀=$(regularizers.βwf) (schedule: $(regularizers.βwf_schedule))")
end
