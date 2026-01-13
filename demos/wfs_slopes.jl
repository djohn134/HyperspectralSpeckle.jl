include("../src/HyperspectralSpeckle.jl");
using Main.HyperspectralSpeckle;
using Statistics;


############# Data Parameters #############
FTYPE = Float32;
# folder = "/home/dan/Desktop/JASS_2024/tests";
folder = "test"
id = "_phaseNoFfm"
verb = true
plot = true
###########################################

##### Size, Timestep, and Wavelengths #####
nsubaps_side = 6
datafile_wfs = "$(folder)/Dr0_20_ISH$(nsubaps_side)x$(nsubaps_side)_images.fits"
images_wfs, nsubaps, ~, wfs_dim, exptime_wfs, times_wfs = readimages(datafile_wfs, FTYPE=FTYPE)
build_dim = 256
nλ = 1
λ_ref = 500.0e-9  # m
λmin = λ_nyquist = 500.0e-9
λmax = 500.0e-9  # m
λ = collect(range(λmin, stop=λmax, length=nλ))  # m
###########################################

### Detector & Observations Parameters ####
D = 3.6  # m
D_inner_frac = 0.0
########## Create Optical System ##########
filter = OpticalElement(name="Bessell:V", FTYPE=FTYPE)
beamsplitter = OpticalElement(λ=[400.0e-9, 1000.0e-9], response=[0.5, 0.5], FTYPE=FTYPE)
######### Create Full-Ap Detector #########
# observations = [observations_full]
optics_wfs = OpticalSystem([filter, beamsplitter], λ, verb=verb, FTYPE=FTYPE)
detector_wfs = Detector(
    λ=λ,
    λ_nyquist=λ_nyquist,
    pitch=10e-6,
    verb=verb,
    label="Wavefront Sensor",
    FTYPE=FTYPE
)
observations_wfs = Observations(
    times_wfs,
    images_wfs,
    optics_wfs,
    detector_wfs,
    D=D,
    D_inner_frac=D_inner_frac,
    nsubaps_side=nsubaps_side,
    # ϕ_static=ϕ_static_wfs,
    build_dim=build_dim,
    verb=verb,
    label="Wavefront Sensor",
    FTYPE=FTYPE
)
###########################################

calculate_wfs_phases!(observations_wfs)
using GLMakie
# slopes = zeros(observations_wfs.nsubaps_side, observations_wfs.nsubaps_side, observations_wfs.nepochs, 2)
# @views for t=1:observations_wfs.nepochs
#     slopes[:, :, t, 1] .= stack2mosaic(observations_wfs.slopes[:, t, 1], observations_wfs.nsubaps_side, observations_wfs.masks.ix)
#     slopes[:, :, t, 2] .= stack2mosaic(observations_wfs.slopes[:, t, 2], observations_wfs.nsubaps_side, observations_wfs.masks.ix)
# end
# writefits(slopes, "test/wf_phase.fits")
writefits(observations_wfs.phase, "test/wf_phase.fits")
