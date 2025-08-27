module HyperspectralSpeckle

const symbol2str = Dict(
    :opd=>"opd", 
    :phase=>"phase", 
    :psf=>"psf",
    :static_phase=>"static_phase", 
    :gaussian=>"gaussian", 
    :mixed=>"noise", 
    :mle=>"mle", 
    :mrl=>"mrl"
)
export symbol2str

## masks
include("masks.jl")
export Masks
export make_ish_masks, make_annular_mask

## imaging
include("imaging.jl")
export create_refraction_operator, create_refraction_adjoint, create_extractor_operator, create_extractor_adjoint, pupil2psf!, position2phase!
export calculate_composite_pupil!, calculate_composite_phase!, calculate_composite_amplitude!
export create_spectral_irradiance_at_aperture!, create_detector_images, create_detector_images!, add_noise!

include("anisoplanatic.jl")
export AnisoplanaticPatches
export get_center, get_center!, create_patch_extractors, create_patch_extractors_adjoint
export change_heights!

## observations
include("observations.jl")
export OpticalElement, OpticalSystem, Detector, Observations
export calculate_wfs_slopes

## object
include("object.jl")
export Object
export mag2flux, template2object, interpolate_object, poly2object, poly2object!, object2poly, object2poly!, fit_background

## atmosphere
include("atmosphere.jl")
export Atmosphere
export create_phase_screens!, calculate_atmosphere_parameters!, calculate_layer_masks!
export get_refraction, refraction_at_layer_pix, refraction_at_detector_pix, layer_scale_factors, layer_nyquist_sampling_mperpix, layer_nyquist_sampling_arcsecperpix, propagate_layers, wind_profile_greenwood, wind_profile_roberts2011, interpolate_phase, calculate_smoothed_opd, composite_r0_to_layers, calculate_coherence_time

## criterion
include("mle.jl")
export loglikelihood_gaussian, loglikelihood_gaussian!
export fg_object_mle, gradient_object_mle_gaussiannoise!, gradient_object_mle_mixednoise!
export fg_phase_mle, fg_phase_mle, gradient_phase_mle_gaussiannoise!, gradient_phase_mle_mixednoise!
export fg_phase_ffm_mle, fg_phase_ffm_mle, gradient_phase_ffm_mle_gaussiannoise!, gradient_phase_ffm_mle_mixednoise!
export fg_opd_ffm_mle, fg_opd_ffm_mle, gradient_opd_ffm_mle_gaussiannoise!, gradient_opd_ffm_mle_mixednoise!

## mrl
# include("mrl.jl")
# export mrl
# export fg_object_mrl, fg_opd_mrl, fg_phase_mrl
# export gradient_object_mrl_gaussiannoise!, gradient_opd_mrl_gaussiannoise!, gradient_phase_mrl_gaussiannoise!

## regularization
include("regularization.jl")
export Regularizers 
export no_reg, tv2_reg, l2_reg, λtv_reg

## plot
include("plot.jl")
export ReconstructionFigures
export plot_object, plot_layers, plot_opd, plot_phase
export update_object_figure, update_layer_figure, update_opd_figure, update_phase_figure, savefig

## utils
include("utils.jl")
export gettype, create_header, writefits, writefile, writeobject, readobject, readfile, readqe, readimages, readmasks, readfits, readspectrum, readtransmission, vega_spectrum, solar_spectrum
export gaussian_kernel, calculate_entropy, calculate_ssim, shift_and_add, fit_plane, crop, smooth_to_rmse!, bartlett_hann2d, super_gaussian, block_reduce!, block_reduce, block_replicate!, block_replicate, stack2mosaic, create_zernike_screen, smooth_to_resolution, interpolate1d, center_of_gravity
export zeros!, ones!, ft, ift, setup_fft, setup_ifft, ConvolutionPlan, Preconvolution, convolve!, CorrelationPlan, Precorrelate, correlate!, setup_autocorr, setup_operator_mul

## reconstruct
include("reconstruct.jl")
export Reconstruction, Helpers, PatchHelpers
export gaussian_weighting, mixed_weighting, reconstruct!, height_solve!
export ConstantSchedule, LinearSchedule, ReciprocalSchedule, ExponentialSchedule

## logos
include("logos.jl")
export show_the_sausage, show_the_satellite

include("prints.jl")

end