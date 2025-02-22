import Interpolations: LinearInterpolation, Line


const ELEMENT_FILENAMES = Dict(
    ## Filter
    "Bessell:U"=>"data/optics/Generic_Bessell.U.dat",
    "Bessell:B"=>"data/optics/Generic_Bessell.B.dat",
    "Bessell:V"=>"data/optics/Generic_Bessell.V.dat",
    "Bessell:R"=>"data/optics/Generic_Bessell.R.dat",
    "Bessell:I"=>"data/optics/Generic_Bessell.I.dat",
    ## Lens Coatings
    "Thorlabs:A"=>"data/optics/thorlabs-A.dat",
    "Thorlabs:B"=>"data/optics/thorlabs-B.dat",
    "Thorlabs:AB"=>"data/optics/thorlabs-AB.dat",
    "Thorlabs:MLA-AR"=>"data/optics/thorlabs-mla-ar.dat",
    ## Mirror Coatings
    "Thorlabs:OAP-P01"=>"data/optics/thorlabs-OAP-45AOI-P01.dat",
    "Thorlabs:Plano-P01"=>"data/optics/thorlabs-plano-45AOI-P01.dat",
    ## Dichroics
    "Thorlabs:DMLP650P-transmitted"=>"data/optics/DMLP650-transmitted.dat",
    "Thorlabs:DMLP650P-reflected"=>"data/optics/DMLP650-reflected.dat",
    "Thorlabs:DMLP805P-transmitted"=>"data/optics/DMLP805-transmitted.dat",
    "Thorlabs:DMLP805P-reflected"=>"data/optics/DMLP805-reflected.dat",
)

abstract type AbstractDetector end
abstract type AbstractOpticalSystem end
abstract type AbstractObservations end
function Base.display(detector::T) where {T<:AbstractDetector}
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

function Base.display(optical_system::T) where {T<:AbstractOpticalSystem}
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Optical system\n"); print(Crayon(reset=true))
    println("\tNumber of elements: $(length(optical_system.elements))")
    println("\tWavelength: $(minimum(optical_system.λ)) — $(maximum(optical_system.λ)) m")
end

function Base.display(observations::T) where {T<:AbstractObservations}
    print(Crayon(underline=true, foreground=(255, 215, 0), reset=true), "Observations\n"); print(Crayon(reset=true))
    println("\tImage Size: $(observations.dim)×$(observations.dim) pixels")
    println("\tNumber of frames: $(observations.nepochs)")
    println("\tNumber of subapertures: $(observations.nsubaps_side)×$(observations.nsubaps_side) subapertures")
    println("\tTelescope Diameter: $(observations.D) m")
    println("\tZenith angle: $(observations.ζ) deg")
end

struct OpticalElement{T<:AbstractFloat}
    name::String
    λ::Vector{T}
    response::Vector{T}
    xflip::Bool
    yflip::Bool
    function OpticalElement(;
            λ=[],
            response=[],
            xflip=false,
            yflip=false,
            name="",
            FTYPE=Float64
        )
        """
            OpticalElement(; λ=..., response=..., xflip=..., yflip=..., name=..., FTYPE=...)

        Creates an `OpticalElement` structure. If `name` is specified and matches an entry in `ELEMENT_FILENAMES`
        it will read the spectral response of the object from the corresponding file. `λ` and `response` can be 
        specified manually to create an optical element.

        * `xflip` can be specified to cause the element to flip the image about the y axis
        * `yflip` can be specified to cause the element to flip the image about the x axis
        * `FTYPE` can be specified to change the Floating-point precision
        """
        if name != ""
            λ₀, response = readfile(ELEMENT_FILENAMES["$name"])
            λ₀ .*= 1e-9
            if λ != []
                itp = LinearInterpolation(λ₀, response, extrapolation_bc=Line())
                response = max.(0.0, itp(λ))
            else
                λ = λ₀
            end
        end

        return new{FTYPE}(name, λ, response, xflip, yflip)
    end
end

struct Detector{T<:AbstractFloat, S<:Real} <: AbstractDetector
    λ::Vector{T}
    λ_nyquist::T
    qe::Vector{T}
    rn::T
    gain::T
    saturation::T
    pixscale::T
    exptime::T
    function Detector(;
            λ=[],
            λ_nyquist=400.0,
            pixscale=0.0,
            qe=[1.0],
            rn=0.0,
            gain=1.0,
            saturation=1e99,
            exptime=5e-3,
            verb=true,
            FTYPE=Float64,
            DTYPE=FTYPE
        )
        detector = new{FTYPE, DTYPE}(λ, λ_nyquist, qe, rn, gain, saturation, pixscale, exptime)
        if verb == true
            display(detector)
        end
        return detector
    end
end

struct OpticalSystem{T<:AbstractFloat} <: AbstractOpticalSystem
    elements::Vector{OpticalElement{T}}
    λ::Vector{T}
    response::Vector{T}
    xflip::Bool
    yflip::Bool
    function OpticalSystem(
            elements,
            λ;
            verb=true,
            FTYPE=Float64
        )

        xflip = false
        yflip = false
        response = ones(FTYPE, length(λ))
        for element in elements
            itp = LinearInterpolation(element.λ, element.response, extrapolation_bc=Line())
            response .*= max.(0.0, itp(λ))
            xflip = xflip ⊻ element.xflip
            yflip = yflip ⊻ element.yflip
        end

        optical_system =  new{FTYPE}(elements, λ, response, xflip, yflip)
        if verb == true
            display(optical_system)
        end
        return optical_system
    end
end

mutable struct Observations{T<:AbstractFloat, S<:Real} <: AbstractObservations
    optics::OpticalSystem{T}
    phase_static::Array{T, 3}
    detector::Detector{T, S}
    ζ::T
    D::T
    times::Vector{T}
    nepochs::Int64
    nsubaps::Int64
    nsubaps_side::Int64
    dim::Int64
    images::Array{S, 4}
    entropy::Matrix{T}
    model_images::Array{T, 4}
    w::Vector{Int64}
    positions::Array{T, 4}
    function Observations(
            optics,
            detector;
            ζ=Inf,
            D=Inf,
            times=[],
            nepochs=0,
            nsubaps=0,
            nsubaps_side=1,
            dim=0,
            ϕ_static=[;;;],
            verb=true,
            FTYPE=Float64
        )
        nepochs = (nepochs==0) ? length(times) : nepochs
        DTYPE = gettypes(detector)[2]
        optics.response .*= detector.qe
        observations = new{FTYPE, DTYPE}(optics, ϕ_static, detector, ζ, D, times, nepochs, nsubaps, nsubaps_side, dim)
        if verb == true
            display(observations)
        end
        return observations
    end
    function Observations(
            times,
            images,
            optics,
            detector;
            ζ=Inf,
            D=Inf,
            nsubaps_side=1,
            ϕ_static=[;;;],
            verb=true,
            FTYPE=Float64
        )
        dim, ~, nsubaps, nepochs = size(images)
        entropy = [calculate_entropy(images[:, :, n, t]) for n=1:nsubaps, t=1:nepochs]
        DTYPE = gettypes(detector)[2]
        optics.response .*= detector.qe
        observations = new{FTYPE, DTYPE}(optics, ϕ_static, detector, ζ, D, times, nepochs, nsubaps, nsubaps_side, dim, images, entropy)
        if verb == true
            display(observations)
        end
        return observations
    end
end

@views function calculate_wfs_slopes(observations_wfs)
    FTYPE = gettype(observations_wfs)
    ~, ~, nsubaps, nepochs = size(observations_wfs.images)
    ∇ϕx = zeros(FTYPE, nsubaps, nepochs)
    ∇ϕy = zeros(FTYPE, nsubaps, nepochs)
    composite_image = dropdims(sum(observations_wfs.images, dims=(3, 4)), dims=(3, 4))
    for n=1:nsubaps
        for t=1:nepochs
            # Δy, Δx = Tuple(argmax(ccorr_psf(composite_image, observations_wfs.images[:, :, n, t])))
            Δx, Δy = center_of_gravity(observations_wfs.images[:, :, n, t])
            ∇ϕx[n, t] = Δx * observations_wfs.D / observations_wfs.nsubaps_side
            ∇ϕy[n, t] = Δy * observations_wfs.D / observations_wfs.nsubaps_side
        end
    end
    return ∇ϕx, ∇ϕy
end
