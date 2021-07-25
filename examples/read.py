#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import lrt_data_reader
import lrt_plots


if __name__ == '__main__':
    dataset = Path('../data/WmuHNL15GeV_NoPileUp_Generic')
    events = [540, 1095, 4332]

    for event in events:
        hits, particles, truth = lrt_data_reader.read_event(dataset, event)

        save = Path(f"output/plots/{event}")
        save.mkdir(
            parents=True,
            exist_ok=True
        )

        # lrt_plots.hit_pair_plot_2d(hits, {0: truth}, save/"truth.png")
        lrt_plots.hit_position_plot_3d_no_group(hits, save/"hit3D.png")
