#!/usr/bin/env julia

import Celeste.ParallelRun: one_node_joint_infer, infer_init, BoundingBox,
                            get_overlapping_fields
import Celeste.SDSSIO: RunCamcolField, load_field_images
import Celeste.Infer: find_neighbors

using Images, Colors, FixedPointNumbers, ImageView
using WCS
using Interpolations


const usage_info =
"""
Usage:
  display-box.jl <ramin> <ramax> <decmin> <decmax>
"""

const datadir = joinpath(Pkg.dir("Celeste"), "test", "data")


function display_box(box::BoundingBox)
    rcfs = get_overlapping_fields(box, datadir)
    @show rcfs

    wd = pwd()
    cd(datadir)
    for rcf in rcfs
        run(`make RUN=$(rcf.run) CAMCOL=$(rcf.camcol) FIELD=$(rcf.field)`)
    end
    cd(wd)

    catalog, target_sources = infer_init(rcfs, datadir; box=box)
    @show length(catalog)
    @show length(target_sources)

    images = load_field_images(rcfs, datadir)
    @show length(images)

#    b_to_symbol = [:u, :g, :r, :i, :z]
#    ugriz = UGRIZ(0., 0., 0., 0., 0.)
#    getfield(ugriz, b_to_symbol[img.b])

    H = 200
    W = ceil(Int64, H * (box.decmax - box.decmin) / (box.ramax - box.ramin))

    coadd = zeros(H, W, 5)

    count = zeros(Int64, H, W, 5)
    for img in images
        nosky = (img.pixels ./ img.iota_vec) - img.epsilon_mat
        itp = interpolate(nosky, BSpline(Linear()), OnCell())

        h0, w0 = world_to_pix(img.wcs, [box.ramin, box.decmin])
        h1, w1 = world_to_pix(img.wcs, [box.ramax, box.decmax])

        # image (the source) dimensions
        h0 = max(h0, 1)
        w0 = max(w0, 1)
        h1 = min(h1, size(img.pixels, 1))
        w1 = min(w1, size(img.pixels, 2))

        # stamp (the target) dimensions
        hh0 = 1 + (h0 - h0)
        ww0 = 1 + (w0 - w0)
        hh1 = hh0 + (h1 - h0)
        ww1 = ww0 + (w1 - w0)

        for w in 1:W, h in 1:H
        coadd[hh0:hh1, ww0:ww1, img.b] += nosky[h0:h1, w0:w1]
    end

    @assert length(images) % 5 == 0
    coadd /= (length(images) / 5)

    coadd -= minimum(coadd)
    coadd /= maximum(coadd)

    for i in 1:length(coadd)
        if isnan(coadd[i])
            coadd[i] = 0.
        end
    end

    coadd = max(0., coadd)
    coadd /= maximum(quantile(coadd[:], .999))
    coadd = min(1., coadd)

    const rgb_weights = [0.0 0.1 0.2 0.3 0.4;
                         0.1 0.2 0.4 0.2 0.1;
                         0.4 0.3 0.2 0.1 0.0]

    coadd_rgb = Array(RGB{U8}, H, W)
    for h in 1:H, w in 1:W
        val1 = rgb_weights * coadd[h, w, :]
        val2 = any(isnan.(val1)) ? zeros(3) : val1
        coadd_rgb[h, w] = RGB{U8}(val2...)
    end

    ImageView.view(coadd_rgb, pixelspacing = [1,1])

    neighbor_map = find_neighbors(target_sources, catalog, images)

    ctni = (catalog, target_sources, neighbor_map, images)
    results = one_node_joint_infer(ctni...; use_fft=true)
    @show length(results)

    readline(STDIN)
end


if length(ARGS) != 4
    println(usage_info)
else
    box = BoundingBox(ARGS...)
    display_box(box)
end
