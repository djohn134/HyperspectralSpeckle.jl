using PerceptualColourMaps
using GLMakie
GLMakie.activate!()

const PLOT_OPTIONS = (
    CMAP_OBJECT = cmap("L01"),
    CMAP_WAVEFRONT = cmap("L20")
)


mutable struct ReconstructionFigures
    object_fig::Figure
    wf_fig::Figure
    static_phase_fig::Figure
    heights_fig::Figure
    object_ax::Axis
    wf_ax::Vector{Axis}
    static_phase_ax::Vector{Axis}
    heights_ax::Axis
    object_obs::Observable
    wf_obs::Vector{Observable}
    static_phase_obs::Vector{Observable}
    heights_obs::Vector{Observable}
    ϵ_obs::Vector{Observable}
    function ReconstructionFigures()
        return new()
    end
end

function plot_object(object; show=true, write=false, filename="", label="Irradiance [ph/s/m^2]")
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], titlesize=18, aspect=DataAspect())
    ax.title = "Object [integrated]"
    hidedecorations!(ax)
    obs = Observable(rotr90(object.Δλ*dropdims(sum(object.object, dims=3), dims=3)))
    plt = plot!(ax, obs, colormap=PLOT_OPTIONS[:CMAP_OBJECT])
    c = Colorbar(fig[1, 2], plt, width=10, height=Relative(1), label=label, labelsize=16, ticklabelsize=16)
    trim!(fig.layout)

    if show == true
        display(GLMakie.Screen(), fig)
    end

    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig, ax, obs
end

function plot_opd(atmosphere; show=false, write=false, filename="", label="OPD [nm]")
    fig = Figure(size=(400*atmosphere.nlayers, 400))
    ax = [Axis(fig[1, 2*l], titlesize=18, aspect=DataAspect()) for l=1:atmosphere.nlayers]
    [ax[l].title = "Layer $(l) - $(label)" for l=1:atmosphere.nlayers]
    hidedecorations!.(ax)
    obs = [Observable(rotr90(atmosphere.opd[:, :, l] .* 1e9)) for l=1:atmosphere.nlayers]
    plt = [plot!(ax[l], obs[l], colormap=PLOT_OPTIONS[:CMAP_WAVEFRONT]) for l=1:atmosphere.nlayers]
    [Colorbar(fig[1, 2*l+1], plt[l], width=10, height=Relative(3/4), labelsize=16, ticklabelsize=16) for l=1:atmosphere.nlayers]
    trim!(fig.layout)
    
    if show == true
        display(GLMakie.Screen(), fig)
    end

    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig, ax, obs
end

function plot_phase(atmosphere; show=false, write=false, filename="", label="Phase [rad] @ $(round(atmosphere.λ[1]*1e9, digits=1)) nm")
    fig = Figure(size=(400*atmosphere.nlayers, 400))
    ax = [Axis(fig[1, 2*l], titlesize=18, aspect=DataAspect()) for l=1:atmosphere.nlayers]
    [ax[l].title = "Layer $(l) - $(label)" for l=1:atmosphere.nlayers]
    hidedecorations!.(ax)
    obs = [Observable(rotr90(atmosphere.phase[:, :, l, 1])) for l=1:atmosphere.nlayers]
    plt = [plot!(ax[l], obs[l], colormap=PLOT_OPTIONS[:CMAP_WAVEFRONT]) for l=1:atmosphere.nlayers]
    [Colorbar(fig[1, 2*l+1], plt[l], width=10, height=Relative(3/4), labelsize=16, ticklabelsize=16) for l=1:atmosphere.nlayers]
    trim!(fig.layout)
    
    if show == true
        display(GLMakie.Screen(), fig)
    end

    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig, ax, obs
end

function plot_static_phase(observations; show=false, write=false, filename="", label="Phase [rad] @ $(round(observations[1].detector.λ[1] * 1e9, digits=1)) nm")
    ndatasets = length(observations)
    fig = Figure(size=(400*ndatasets, 400))
    ax = [Axis(fig[1, 2*dd], titlesize=18, aspect=DataAspect()) for dd=1:ndatasets]
    [ax[dd].title = "Channel $(dd) - $(label)" for dd=1:ndatasets]
    hidedecorations!.(ax)
    obs = [Observable(rotr90(observations[dd].phase_static[:, :, 1])) for dd=1:ndatasets]
    plt = [plot!(ax[dd], obs[dd], colormap=PLOT_OPTIONS[:CMAP_WAVEFRONT]) for dd=1:ndatasets]
    [Colorbar(fig[1, 2*dd+1], plt[dd], width=10, height=Relative(3/4), labelsize=16, ticklabelsize=16) for dd=1:ndatasets]
    trim!(fig.layout)
    
    if show == true
        display(GLMakie.Screen(), fig)
    end

    if write == true
        save(filename, fig, px_per_unit=2)
        println("Figure written to $(filename)")
    end
    return fig, ax, obs
end

function plot_heights(atmosphere; heights=[], ϵ=[], show=false)
    FTYPE = gettype(atmosphere)
    if (heights == []) || (ϵ == [])
        heights = [FTYPE[] for ~=1:atmosphere.nlayers-1]
        ϵ = [FTYPE[] for ~=1:atmosphere.nlayers-1]
    end
    ϵ_real = cat(ϵ..., dims=1)
    ϵ_real = ϵ_real[.!(isnan.(ϵ_real))]
    base10exp = (length(ϵ_real)==0) ? 0 : floor(Int64, log10(minimum(ϵ_real)))
    ylabel = rich("Criterion [×10", superscript(string(base10exp)) , "]")
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], xlabel="Height [m]", ylabel=ylabel, xlabelsize=18, xticklabelsize=16, ylabelsize=18, yticklabelsize=16)
    heights_obs = [Observable(heights[l]) for l=1:atmosphere.nlayers-1]
    ϵ_obs = [Observable(ϵ[l] ./ 10^base10exp) for l=1:atmosphere.nlayers-1]
    for l=1:atmosphere.nlayers-1
        scatterlines!(ax, heights_obs[l], ϵ_obs[l])
    end

    if show == true
        display(GLMakie.Screen(), fig)
    end
    return fig, ax, heights_obs, ϵ_obs
end

function update_object_figure(object, reconstruction)
    figs = reconstruction.figures
    figs.object_obs[] = rotr90(object)
end

function update_opd_figure(opd, atmosphere, reconstruction)
    figs = reconstruction.figures
    for l=1:atmosphere.nlayers
        figs.wf_obs[l][] = rotr90(opd[:, :, l] .* 1e9)
        reset_limits!(figs.wf_ax[l])
    end
end

function update_phase_figure(phase, atmosphere, reconstruction)
    figs = reconstruction.figures
    for l=1:atmosphere.nlayers
        figs.wf_obs[l][] = rotr90(phase[:, :, l, 1])
        reset_limits!(figs.wf_ax[l])
    end
end

function update_static_phase_figure(phase, observations, reconstruction)
    figs = reconstruction.figures
    for dd=1:reconstruction.ndatasets
        figs.static_phase_obs[dd][] = rotr90(phase[:, :, 1])
        reset_limits!(figs.static_phase_ax[dd])
    end
end

function update_layer_figure(atmosphere, reconstruction)
    getfield(Main, Symbol("update_$(symbol2str[reconstruction.wavefront_parameter])_figure"))(getfield(atmosphere, reconstruction.wavefront_parameter), atmosphere, reconstruction)
end

function update_heights_figure(heights, ϵ, atmosphere, reconstruction)
    ϵ_real = cat(ϵ..., dims=1)
    ϵ_real = ϵ_real[.!(isnan.(ϵ_real))]
    base10exp = (length(ϵ_real)==0) ? 0 : floor(Int64, log10(minimum(ϵ_real)))
    ylabel = rich("Criterion [×10", superscript(string(base10exp)) , "]")
    figs = reconstruction.figures
    figs.heights_ax.ylabel = ylabel
        for l=1:atmosphere.nlayers-1
            figs.heights_obs[l][] = heights[l]
            figs.ϵ_obs[l][] = ϵ[l] ./ 10^base10exp
            reset_limits!(figs.heights_ax)
        end
end

function savefig(filename, fig, px_per_unit)
    save(filename, fig, px_per_unit=px_per_unit)
end
