include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;


############# Data Parameters #############
FTYPE = Float32;
<<<<<<< HEAD
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "data/test"
id = "_aniso"
verb = true
plot = true
=======
folder = "/home/dan/Desktop/dissertation";
verb = true
plot = false
>>>>>>> main
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64
nsubaps_side = 6
nλ₀ = 1
nλ = 1
nλint = 1
<<<<<<< HEAD
λ_ref = 500.0e-9
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9
=======
λ_nyquist = 500.0
λ_ref = 500.0
λmin = 500.0
λmax = 500.0
>>>>>>> main
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
λ₀ = (nλ₀ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ₀))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
resolution = mean(λ) / Δλ
###########################################
id = "_height_solve"

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
isoplanatic = false
patch_overlap = 0.5
patch_dim = 128
<<<<<<< HEAD
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE, verb=verb)
=======
##### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, patch_overlap, isoplanatic=isoplanatic, FTYPE=FTYPE)
>>>>>>> main
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
<<<<<<< HEAD
fov = 50.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
=======
# fov = 20 * 256 / (132 * (256 / 512))
fov = 30.0
pixscale_full = fov / image_dim  # 0.25 .* ((λ .* 1e-9) .* 1e6) ./ D
pixscale_wfs = pixscale_full * nsubaps_side
>>>>>>> main
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
# qe = ones(FTYPE, nλ)
rn = 2.0
exptime = 5e-3
ζ = 0.0
<<<<<<< HEAD
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
# filter = OpticalElement(λ=[0.0, 10000.0], response=[1.0, 1.0], FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[0.0, 10000.0], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
# optics_full = OpticalSystem([filter], λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
=======
######### Create Detector object ##########
filter = Filter(filtername="Bessell:V", FTYPE=FTYPE)
>>>>>>> main
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
<<<<<<< HEAD
    verb=verb,
=======
    filter=filter,
>>>>>>> main
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
datafile = "$(folder)/Dr0_20_ISH1x1_images.fits"
images_full, ~, nepochs, image_dim, exptime_full, times_full = readimages(datafile, FTYPE=FTYPE)
ϕ_static_full = zeros(FTYPE, image_dim, image_dim, nλ)
observations_full = Observations(
    times_full,
    images_full,
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    ϕ_static=ϕ_static_full,
    verb=verb,
    FTYPE=FTYPE
)
<<<<<<< HEAD
observations = [observations_full]
# optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
# detector_wfs = Detector(
#     qe=qe,
#     rn=rn,
#     pixscale=pixscale_wfs,
#     λ=λ,
#     λ_nyquist=λ_nyquist,
#     exptime=exptime,
#     verb=verb,
#     FTYPE=FTYPE
# )
# ### Create Full-Ap Observations object ####
# datafile = "$(folder)/Dr0_20_ISH$(nsubaps_side)x$(nsubaps_side)_images.fits"
# images_wfs, nsubaps, ~, wfs_dim = readimages(datafile, FTYPE=FTYPE)
# ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
# observations_wfs = Observations(
#     images_wfs,
#     optics_wfs,
#     detector_wfs,
#     ζ=ζ,
#     D=D,
#     nsubaps_side=nsubaps_side,
#     ϕ_static=ϕ_static_wfs,
#     verb=verb,
#     FTYPE=FTYPE
# )
# observations = [observations_wfs, observations_full]
background_flux = 0.0 * mean(fit_background(observations_full))
=======
##### Create WFS Observations object ######
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_wfs,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    filter=filter,
    FTYPE=FTYPE
)
datafile = "$(folder)/Dr0_20_ISH6x6_images.fits"
observations_wfs = Observations(
    detector_wfs,
    ζ=ζ,
    D=D,
    α=0.5,
    datafile=datafile,
    FTYPE=FTYPE
)
observations = [observations_full]
>>>>>>> main
###########################################

########## Create Full-Ap Masks ###########
masks_full = Masks(
    image_dim,
    λ,
    nsubaps_side=1,  
    λ_nyquist=λ_nyquist, 
    verb=verb,
    FTYPE=FTYPE
)
<<<<<<< HEAD
=======
############ Create WFS Masks #############
masks_wfs = Masks(
    dim=observations_full.dim,
    nsubaps_side=nsubaps_side, 
    λ=λ, 
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
>>>>>>> main
masks = [masks_full]
# masks_wfs = Masks(
#     image_dim,
#     λ,
#     nsubaps_side=nsubaps_side,
#     λ_nyquist=λ_nyquist, 
#     verb=verb,
#     FTYPE=FTYPE
# )
# nsubaps = masks_wfs.nsubaps
# masks_wfs.scale_psfs = masks_full.scale_psfs
# masks = [masks_wfs, masks_full]
###########################################

############ Object Parameters ############
<<<<<<< HEAD
object_range = 300.0e3  # km
=======
object_height = 550.0# 1.434e6  # km
>>>>>>> main
############## Create object ##############
# object_arr = max.(dropdims(mean(observations_full.images, dims=(3, 4)), dims=(3, 4)) .- background_flux/observations_full.dim^2, 0)
# object_arr = repeat(object_arr, 1, 1, nλ)
# object_arr ./= sum(object_arr)
# object_arr .*= mean(sum(observations_full.images, dims=(1, 2, 3))) - background_flux
object_arr = readfits("$(folder)/object_recon.fits")
# object_arr = readfits("$(folder)/object_truth.fits")
# object_arr = zeros(FTYPE, image_dim, image_dim, nλ)
~, spectrum = solar_spectrum(λ=λ)
object = Object(
    object_arr,
    λ=λ,
    object_range=object_range, 
    fov=fov,
    dim=observations_full.dim,
    background_flux=background_flux,
    spectrum=spectrum,
    scaled=true,
    FTYPE=FTYPE,
    verb=verb
)
<<<<<<< HEAD
=======

object.object = readfits("$(folder)/object_recon_iso.fits")
>>>>>>> main
###########################################


########## Atmosphere Parameters ##########
heights = [0.0, 7000.0, 12500.0]
wind_speed = wind_profile_roberts2011(heights, ζ)
heights .*= 0.0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λ
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
<<<<<<< HEAD
~, transmission = readtransmission("data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
# transmission = ones(FTYPE, nλ)
=======
# maskfile = "$(folder)/layer_masks.fits"
>>>>>>> main
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations, 
    masks,
    object, 
    patches,
    wind=wind, 
    heights=heights,
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
<<<<<<< HEAD
    create_screens=false,
    FTYPE=FTYPE,
    verb=verb
)
######### Set phase screens start #########
# atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
atmosphere.phase = readfits("$(folder)/phase_recon.fits")
=======
    FTYPE=FTYPE
)
########## Create phase screens ###########
calculate_screen_size!(atmosphere, observations_full, object, patches)
calculate_pupil_positions!(atmosphere, observations_full)
calculate_layer_masks_eff!(patches, atmosphere, observations_full, object, masks_full)
atmosphere.opd = readfits("$(folder)/opd_recon_iso.fits")
>>>>>>> main
###########################################

######### Reconstruction Object ###########
reconstruction = Reconstruction(
    atmosphere,
    observations,
    object,
    patches,
    λmin=λmin,
    λmax=λmax,
    nλ=nλ,
    nλint=nλint,
    niter_mfbd=1,
    maxiter=500,
    # indx_boot=[1:2],
<<<<<<< HEAD
    wavefront_parameter=:phase,
    minimization_scheme=:mle,
    noise_model=:mixed,
    maxeval=Dict("wf"=>1000, "object"=>1000),
    smoothing=true,
    fwhm_schedule=ConstantSchedule(0.5),
    build_dim=image_dim,
    grtol=1e-2,
    verb=verb,
    plot=plot,
=======
    # weight_function=gaussian_weighting,
    # gradient_object=gradient_object_gaussiannoise!,
    # gradient_opd=gradient_opd_gaussiannoise!,
    maxeval=Dict("opd"=>1000, "object"=>1000),
    smoothing=false,
    grtol=1e-2,
    build_dim=image_dim,
    verb=verb,
    plot=plot,
    mfbd_verb_level="silent",
>>>>>>> main
    FTYPE=FTYPE
);
###########################################

hmin = [5000.0, 10000.0]  # m
hmax = [10000.0, 20000.0]  # m
hstep = [1000.0, 1000.0]  # m
niters = 2
<<<<<<< HEAD
ϵ_heights, height_trials = height_solve!(observations, atmosphere, object, patches, masks, reconstruction, hmin=hmin, hmax=hmax, hstep=hstep, niters=niters)
writefile(cat(height_trials, dims=1), cat(ϵ_heights, dims=1), "$(folder)/heights$(id).dat")

using GLMakie
fig = Figure(size=(400, 400))
ax = Axis(fig[1, 1], xlabel="Height [m]", ylabel="ϵ")
for l=1:atmosphere.nlayers-1
    lines!(ax, height_trials[l], ϵ_heights[l])
end
save("$(folder)/heights$(id).png", fig, px_per_unit=16)

## Write anisoplanatic phases and images ##
writefile(reconstruction.ϵ, "$(folder)/recon$(id).dat")
writefits(object.object, "$(folder)/object_recon$(id).fits")
writefits(atmosphere.phase, "$(folder)/phase_recon$(id).fits")
writefits(observations_full.model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
res_full = observations_full.model_images .- observations_full.images
writefits(res_full, "$(folder)/residuals_ISH1x1_recon$(id).fits")
=======
ϵ_heights, height_trials, atmosphere, object = height_solve!(observations, atmosphere, object, patches, masks, reconstruction, hmin=hmin, hmax=hmax, hstep=hstep, niters=niters)
writefile(cat(height_trials..., dims=1), cat(ϵ_heights..., dims=1), "$(folder)/heights$(id).dat")
savefig("$(folder)/heights$(id).png", reconstruction.figures.heights_fig, 2)

res_full = observations_full.model_images .- observations_full.images
# res_wfs = observations_wfs.model_images .- observations_wfs.images

atmosphere.opd .*= atmosphere.masks[:, :, :, 1]
writefits(atmosphere.opd, "$(folder)/opd_recon$(id).fits")
# writefits(patches.ϕ_composite, "$(folder)/phase_composite_recon$(id).fits")
# for l=1:atmosphere.nlayers
#     atmosphere.opd[findall(atmosphere.masks[:, :, l, 1] .> 0), l] .-= mean(atmosphere.opd[findall(atmosphere.masks[:, :, l, 1] .> 0), l])
#     atmosphere.opd[:, :, l] .-= fit_plane(atmosphere.opd[:, :, l], atmosphere.masks[:, :, l, 1])
# end
# calculate_composite_pupil_eff(patches, atmosphere, observations, object, masks, build_dim=image_dim, propagate=false)
# writefits(atmosphere.opd, "$(folder)/opd_recon_woplane$(id).fits")
# writefits(patches.ϕ_composite, "$(folder)/phase_composite_recon_woplanes$(id).fits")
writefits(object.object, "$(folder)/object_recon$(id).fits")
writefits(observations_full.model_images, "$(folder)/models_ISH1x1_recon$(id).fits")
# writefits(observations_wfs.model_images, "$(folder)/models_ISH6x6_recon$(id).fits")
# writefits(patches.psfs[end], "$(folder)/psfs_ISH1x1_recon$(id).fits")
# writefits(patches.psfs[1], "$(folder)/psfs_ISH6x6_recon$(id).fits")
writefits(res_full, "$(folder)/residuals_ISH1x1_recon$(id).fits")
# writefits(res_wfs, "$(folder)/residuals_ISH6x6_recon$(id).fits")
>>>>>>> main
###########################################
