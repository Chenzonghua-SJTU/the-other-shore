"""Plot clean example trajectories, not noisy agent observations or scores.

Requires optional matplotlib. Run from package root:
    python examples/plot_worlds.py --output validation/world-atlas.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from development_worlds import catalog
from development_worlds.worlds import DevelopmentSimulator
from development_worlds.world_specs import WORLD_SPECS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation/world-atlas.png'))
    args = parser.parse_args()
    fig, axes = plt.subplots(4, 5, figsize=(17, 14), facecolor='#f5f6f9')
    colors = ['#146c94', '#e57c23', '#8b5fbf', '#24875c']
    for world, ax in zip(catalog.WORLDS, axes.flat):
        sim = DevelopmentSimulator(world)
        case = catalog.validation_cases(world)[2]
        probes = case['probes']
        history = sim.integrate([p['pos'] for p in probes], [p['vel'] for p in probes],
            t_launch=case['t_launch'], duration=case['readout'][-1],
            source_properties=case['source_properties'], probe_properties=[p['properties'] for p in probes])
        pos = history['positions']
        for i, color in enumerate(colors):
            ax.plot(pos[::10, i, 0], pos[::10, i, 1], color=color, lw=1.1, alpha=.9)
            ax.scatter(*pos[0, i], color=color, s=18, zorder=3)
        background = sim.background_initial_state()
        ax.scatter(*background['positions'][0], color='#232938', marker='+', s=30)
        if world == '14':
            ax.scatter(*background['positions'][1], color='#232938', marker='+', s=30)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.14)
        ax.tick_params(labelsize=7)
        name = WORLD_SPECS[world]['mechanism'].replace('_', ' ')
        ax.set_title(f'{int(world):02d}  {name}', fontsize=9.5, loc='left', pad=8)
        ax.text(.03, .03, f"t = {case['t_launch']:g} .. {case['t_launch']+case['readout'][-1]:g}",
                transform=ax.transAxes, fontsize=7, color='#566173')
        for spine in ax.spines.values():
            spine.set_color('#d2d8e1')
    fig.suptitle('SHORE / 20 PHYSICS WORLDS', x=.05, ha='left', fontsize=24, weight='bold', color='#182840')
    fig.text(.05, .945, 'Original worlds 01-06 + synthetic training worlds 07-20 | Four passive probes per panel', fontsize=11, color='#566173')
    fig.text(.05, .025, 'Clean simulator trajectories at fixed dt=0.005. Dots mark launch positions; axes vary by panel. This is not an Agent performance result.', fontsize=10, color='#566173')
    fig.subplots_adjust(left=.05, right=.98, top=.91, bottom=.07, hspace=.3, wspace=.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, facecolor=fig.get_facecolor())
    print(args.output.resolve())


if __name__ == '__main__':
    main()
