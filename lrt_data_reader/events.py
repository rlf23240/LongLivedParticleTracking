#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def read_event(dataset, event):
    """
    :param dataset: Name of dataset.
    :param event: Event ID.
    :return: Three DataFrame: hits, particles and truth labels.
    """
    dataset = Path(dataset)

    print(f"Reading {dataset}/{event}...")

    data = torch.load(dataset / f'{event}', map_location='cpu')

    hit_ids = data.hid.numpy()
    hit_positions = data.x.numpy()
    hit_layers = data.layers.numpy()
    true_edges = data.layerless_true_edges.transpose(0, 1).numpy()
    particle_ids = data.pid.numpy()

    hits = _construct_hit_dataframe(
        event, hit_ids, hit_layers, hit_positions
    )

    particles = _construct_particle_id_dataframe(
        hit_ids, particle_ids
    )

    truth = _construct_true_edge_dataframe(
        hits, true_edges
    )

    return hits, particles, truth


def _construct_hit_dataframe(event_id, hit_ids, hit_layers, hit_positions):
    """
    Construct hit dataframe from corresponding raw numpy array.

    :param hit_ids: Hit identifiers.
    :param hit_layers: Hit layers.
    :param hit_positions: Hit position. In (r, phi, z) coordinate.
    :return: Hit dataframe.
    """
    hits = pd.DataFrame()
    hits['evtid'] = np.full(len(hit_ids), event_id)
    hits['hit_id'] = hit_ids
    hits['layer_id'] = hit_layers
    hits['r'] = hit_positions[:, 0]
    hits['phi'] = hit_positions[:, 1]
    hits['z'] = hit_positions[:, 2]

    # Cylindrical coord.
    r = hits['r']
    phi = hits['phi']
    z = hits['z']

    # Compute cartesian coord
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # eta phi space.
    theta = np.arctan2(r, z)
    eta = -np.log(np.tan(theta / 2.0))

    hits = hits.assign(
        x=x,
        y=y,
        theta=theta,
        eta=eta
    )

    return hits


def _construct_particle_id_dataframe(hit_ids, particle_ids):
    """
    Construct hit ID to particle ID lookup table.

    Note that since particle ID is one of result we want to identify,
    we do not merge it to hit dataframe but put it into separated dataframe instead.

    :return: Particle ID dataframe.
    """
    particles = pd.DataFrame()

    particles['hit_id'] = hit_ids
    particles['particle_id'] = particle_ids

    return particles


def _construct_true_edge_dataframe(hits, true_edges):
    """
    Construct true edge from raw data.

    Note that this is ground truth associated with event,
    not construct from particle ID.
    If you seek the truth construct from particle ID,
    see *EdgeFilter* in pairing algorithm.

    :param hits:
        Hit dataframe.
        Must at least contain hit ID.
        Other columns are optional but will be encoded into final dataframe.
    :param true_edges: Raw truth label.
    :return: True edge dataframe.
    """
    edges = pd.DataFrame(columns=[
        'hit_index_1', 'hit_index_2'
    ], data=true_edges)

    # Merge first node.
    edges = pd.merge(
        edges,
        hits.reset_index(),
        how='inner',
        left_on='hit_index_1',
        right_on='index'
    )

    # Merge second node.
    edges = pd.merge(
        edges,
        hits.reset_index(),
        how='inner',
        left_on='hit_index_2',
        right_on='index',
        suffixes=('_1', '_2')
    )

    return edges
