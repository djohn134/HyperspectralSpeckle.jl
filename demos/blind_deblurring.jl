include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;
# using LuckyImaging;
# show_the_sausage()


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "data/test"
id = "_iso"
verb = true
plot = false
###########################################

##### Size, Timestep, and Wavelengths #####
image_dim = 256
wfs_dim = 64
nsubaps_side = 6
nλ = 1
nλint = 1
λ_ref = 500.0
λmin = λ_nyquist = 500.0
λmax = 500.0
λ = (nλ == 1) ? [mean([λmax, λmin])] : collect(range(λmin, stop=λmax, length=nλ))
Δλ = (nλ == 1) ? 1.0 : (λmax - λmin) / (nλ - 1)
resolution = mean(λ) / Δλ
###########################################

########## Anisopatch Parameters ##########
## Unused but sets the size of the layer ##
isoplanatic = true
patch_overlap = 0.5
patch_dim = 64
###### Create Anisoplanatic Patches #######
patches = AnisoplanaticPatches(patch_dim, image_dim, isoplanatic=isoplanatic, FTYPE=FTYPE)
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
fov = 20.0
pixscale_full = fov / image_dim
pixscale_wfs = pixscale_full .* nsubaps_side
qefile = "data/qe/prime-95b_qe.dat"
~, qe = readqe(qefile, λ=λ)
rn = 2.0
exptime = 5e-3
ζ = 0.0
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[0.0, 10000.0], response=[0.5, 0.5], FTYPE=FTYPE)
optics_full = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
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
# ϕ_static_full = repeat(masks_full.masks[:, :, 1, 1] .* create_zernike_screen(image_dim, image_dim÷4, 4, 4.0, FTYPE=FTYPE), 1, 1, nλ)
# writefits(ϕ_static_full, "$(folder)/defocus4.fits")
# ϕ_static_full = readfits("$(folder)/defocus4.fits")
datafile = "$(folder)/Dr0_20_ISH1x1_images.fits"
images_full, ~, ~, image_dim = readimages(datafile, FTYPE=FTYPE)
ϕ_static_full = zeros(FTYPE, image_dim, image_dim, nλ)
observations_full = Observations(
    images_full,
    optics_full,
    detector_full,
    ζ=ζ,
    D=D,
    ϕ_static=ϕ_static_full,
    verb=verb,
    FTYPE=FTYPE
)
# observations = [observations_full]
optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
detector_wfs = Detector(
    qe=qe,
    rn=rn,
    pixscale=pixscale_wfs,
    λ=λ,
    λ_nyquist=λ_nyquist,
    exptime=exptime,
    verb=verb,
    FTYPE=FTYPE
)
### Create Full-Ap Observations object ####
datafile = "$(folder)/Dr0_20_ISH$(nsubaps_side)x$(nsubaps_side)_images.fits"
images_wfs, nsubaps, ~, wfs_dim = readimages(datafile, FTYPE=FTYPE)
ϕ_static_wfs = zeros(FTYPE, image_dim, image_dim, nλ)
observations_wfs = Observations(
    images_wfs,
    optics_wfs,
    detector_wfs,
    ζ=ζ,
    D=D,
    nsubaps_side=nsubaps_side,
    ϕ_static=ϕ_static_wfs,
    verb=verb,
    FTYPE=FTYPE
)
observations = [observations_wfs, observations_full]
background_flux = mean(fit_background.(observations))
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
# masks = [masks_full]
masks_wfs = Masks(
    image_dim,
    λ,
    nsubaps_side=nsubaps_side,
    λ_nyquist=λ_nyquist, 
    FTYPE=FTYPE
)
nsubaps = masks_wfs.nsubaps
masks_wfs.scale_psfs = masks_full.scale_psfs
masks = [masks_wfs, masks_full]
###########################################

############ Object Parameters ############
object_height = 515.0  # km
############## Create object ##############
all_subap_images = max.(dropdims(mean(observations_full.images, dims=(3, 4)), dims=(3, 4)) .- background_flux/observations_full.dim^2, 0)
object_arr = repeat(all_subap_images, 1, 1, nλ)
object_arr ./= sum(object_arr)
object_arr .*= mean(sum(observations_full.images, dims=(1, 2, 3))) - background_flux
# object.object = readfits("$(folder)/object_recon.fits")
# object.object = readfits("$(folder)/object_truth.fits")
# object.object = zeros(FTYPE, image_dim, image_dim, nλ)
object = Object(
    object_arr,
    λ=λ,
    height=object_height, 
    fov=fov,
    dim=observations_full.dim,
    background_flux=background_flux,
    FTYPE=FTYPE
)
###########################################


########## Atmosphere Parameters ##########
heights = [0.0, 7.0, 12.5]
wind_speed = wind_profile_roberts2011(heights, ζ)
# heights .*= 0.0
wind_direction = [45.0, 125.0, 135.0]
wind = [wind_speed wind_direction]
nlayers = length(heights)
scaleby_wavelength = λ_nyquist ./ λ
sampling_nyquist_mperpix = layer_nyquist_sampling_mperpix(D, image_dim, nlayers)
sampling_nyquist_arcsecperpix = layer_nyquist_sampling_arcsecperpix(D, fov, heights, image_dim)
# ~, transmission = readtransmission("data/atmospheric_transmission.dat", resolution=resolution, λ=λ)
transmission = ones(FTYPE, nλ)
############ Create Atmosphere ############
atmosphere = Atmosphere(
    λ,
    observations_full, 
    masks_full,
    object, 
    patches,
    wind=wind, 
    heights=heights,
    sampling_nyquist_mperpix=sampling_nyquist_mperpix,
    sampling_nyquist_arcsecperpix=sampling_nyquist_arcsecperpix,
    λ_nyquist=λ_nyquist,
    λ_ref=λ_ref,
    create_screens=false,
    FTYPE=FTYPE
)
######### Set phase screens start #########
atmosphere.phase = zeros(FTYPE, atmosphere.dim, atmosphere.dim, atmosphere.nlayers, atmosphere.nλ)
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
    niter_mfbd=10,
    maxiter=10,
    # indx_boot=[1:2],
    wavefront_parameter=:phase,
    minimization_scheme=:mle,
    noise_model=:gaussian,
    maxeval=Dict("wf"=>1000, "object"=>1000),
    smoothing=true,
    # fwhm_schedule=ConstantSchedule(0.5),
    grtol=1e-9,
    frtol=1e-9,
    xrtol=1e-9,
    build_dim=image_dim,
    verb=verb,
    plot=plot,
    FTYPE=FTYPE
);
reconstruct!(reconstruction, observations, atmosphere, object, masks, patches, write=false, folder=folder, id=id)
###########################################

###########################################
## Write isoplanatic phases and images ####
[writefits(observations[dd].model_images, "$(folder)/models_ISH$(observations[dd].nsubaps_side)x$(observations[dd].nsubaps_side)_recon$(id).fits") for dd=1:reconstruction.ndatasets]
writefits(object.object, "$(folder)/object_recon$(id).fits")
writefits(getfield(atmosphere, reconstruction.wavefront_parameter), "$(folder)/$(symbol2str[reconstruction.wavefront_parameter])_recon$(id).fits")
writefile([reconstruction.ϵ], "$(folder)/recon$(id).dat")
###########################################