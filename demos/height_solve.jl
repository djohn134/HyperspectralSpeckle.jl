include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "data/test"
id = "_aniso"
verb = true
plot = true
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64
wfs_dim = 64
nsubaps_side = 6
nλ₀ = 1
nλ = 1
nλint = 1
λ_ref = 500.0e-9
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
λ₀ = (nλ₀ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ₀))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
resolution = mean(λ) / Δλ
###########################################
id = "_height_solve"

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
## Unused but sets the size of the layer ##
isoplanatic = false
patch_overlap = 0.5
patch_dim = 128
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE, verb=verb)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 50.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
# qe = ones(FTYPE, nλ)
rn = 2.0
rn = 2.0
exptime = 5e-3
ζ = 0.0
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
# filter = OpticalElement(λ=[0.0, 10000.0], response=[1.0, 1.0], FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[0.0, 10000.0], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
# optics_full = OpticalSystem([filter], λ, verb=verb, FTYPE=FTYPE)
######### Create Full-Ap Detector #########
detector_full = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_full,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
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
object_range = 300.0e3  # km
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
###########################################


########## Atmosphere Parameters ##########
heights = [0.0, 7000.0, 12500.0]
wind_speed = wind_profile_roberts2011(heights, ζ)
heights .*= 0.0
heights .*= 0.0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λ
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
~, transmission = readtransmission("data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
# transmission = ones(FTYPE, nλ)
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
    create_screens=false,
    FTYPE=FTYPE,
    verb=verb
)
######### Set phase screens start #########
# atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
atmosphere.phase = readfits("$(folder)/phase_recon.fits")
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
    λmin=λmin,
    λmax=λmax,
    nλ=nλ,
    nλint=nλint,
    niter_mfbd=1,
    maxiter=500,
    # indx_boot=[1:2],
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
    FTYPE=FTYPE
);
###########################################

hmin = [5000.0, 10000.0]  # m
hmax = [10000.0, 20000.0]  # m
hstep = [1000.0, 1000.0]  # m
niters = 2
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
###########################################
