from pathlib import Path
import base64
import csv
import html
import io
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

root = Path(__file__).resolve().parent
fit = root / 'validated'
data = json.loads((fit / 'comparison.json').read_text())
data['backend'] = 'Existing tag_mapping.graph.build_graph and tag_mapping.solver.solve, with rank-check and linear-column-scaling fixes'
(fit / 'comparison.json').write_text(json.dumps(data, indent=2) + '\n')
results = json.loads((fit / 'results.json').read_text())
rows = data['rows']
timing = data['timing']
sensitivity = []
for name, scale in [('sensitivity_half', 0.5), ('validated', 1.0), ('sensitivity_double', 2.0)]:
    run = json.loads((root / name / 'comparison.json').read_text())
    assert run['solver_status'].startswith('converged')
    sensitivity.append(dict(odometry_sigma_scale=scale,
                            tag0_error_m=run['rows'][0]['after_error_m'],
                            tag2_error_m=run['rows'][2]['after_error_m'],
                            held_out_rmse_m=run['independent_evaluation']['optimized']['horizontal_rmse_m']))
(root / 'sensitivity.json').write_text(json.dumps(sensitivity, indent=2) + '\n')
shutil.copyfile(fit / 'comparison.csv', root / 'comparison.csv')

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax, bars) = plt.subplots(1, 2, figsize=(12, 6.7), gridspec_kw={'width_ratios': [1.1, 1]})
orange, teal = '#d87916', '#008477'
for r in rows:
    gx, gy = r['ground_truth_x_m'], r['ground_truth_y_m']
    ax.scatter(gy, gx, marker='x', s=110, color='#192c40', linewidths=2, zorder=5)
    ax.annotate('Tag %d%s' % (r['tag_id'], ' (anchor)' if r['anchor'] else ''),
                (gy, gx), xytext=(0, 13), textcoords='offset points', ha='center', fontsize=10,
                bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 1})
    if 'before_x_m' in r:
        ax.plot([r['before_y_m'], r['after_y_m']], [r['before_x_m'], r['after_x_m']], color='#a8b2bc', lw=1)
        ax.scatter(r['before_y_m'], r['before_x_m'], marker='o', s=75, color=orange, zorder=3)
        ax.scatter(r['after_y_m'], r['after_x_m'], marker='s', s=55, color=teal, zorder=4)
    else:
        ax.annotate('no accepted sightings', (gy, gx), xytext=(0, -21), textcoords='offset points', ha='center', fontsize=9)
ax.scatter(0, 0, marker='*', s=140, color='#192c40')
ax.annotate('Start (0, 0)', (0, 0), xytext=(12, 2), textcoords='offset points')
ax.set(xlabel='Y right (m)', ylabel='X forward (m)', title='Tag positions relative to start', ylim=(-.15, 3.1), xlim=(-1.18, 1.12))
ax.set_aspect('equal')
ax.grid(alpha=.2)
ax.legend(handles=[Line2D([], [], marker='x', color='#192c40', lw=0, label='Ground truth'),
                   Line2D([], [], marker='o', color=orange, lw=0, label='Mapper before optimisation'),
                   Line2D([], [], marker='s', color=teal, lw=0, label='After optimisation')],
          loc='lower center', bbox_to_anchor=(.5, -.32), frameon=False, fontsize=10)
known = rows[:3]
indices = np.arange(3)
for offset, key, label, colour in [(-.18, 'before_error_m', 'Before', orange), (.18, 'after_error_m', 'After', teal)]:
    values = [r[key] * 100 for r in known]
    rectangles = bars.bar(indices + offset, values, .34, label=label, color=colour)
    for rect, value in zip(rectangles, values):
        bars.text(rect.get_x() + rect.get_width()/2, value + .65,
                  '<0.1' if value < .1 else f'{value:.1f}', ha='center', fontsize=10)
bars.set(xticks=indices, xticklabels=['Tag 0\n7 sightings', 'Tag 1 (anchor)\n151 sightings', 'Tag 2 (tentative)\n2 sightings'],
         ylabel='Horizontal position error (cm)', title='Error against supplied ground truth', ylim=(0, 36))
bars.legend(frameon=False)
bars.grid(axis='y', alpha=.2)
bars.set_axisbelow(True)
fig.suptitle('Pose graph optimisation using the existing Python mapper', fontsize=17, fontweight='bold', y=.98)
fig.text(.5, .04, 'Approximate replay: unverified exposure timing and floor datum. Tag 1 is constrained and excluded from independent error metrics.', ha='center', fontsize=9, color='#45556a')
fig.subplots_adjust(bottom=.24, top=.86, wspace=.35)
fig.savefig(root / 'comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)

def f(value, precision=3):
    return f'{value:.{precision}f}'

def xy(row, stage):
    return f"({row[stage+'_x_m']:.3f}, {row[stage+'_y_m']:+.3f})"

table = ''
raw_table = ''
for r in rows:
    label = str(r['tag_id']) + (' (anchor)' if r['anchor'] else ' (tentative)' if r['tag_id'] == 2 else '')
    gt = f"({r['ground_truth_x_m']:.3f}, {r['ground_truth_y_m']:+.3f})"
    table += f'<tr><td>{label}</td><td>{r["accepted_frames"]}</td><td>{gt}</td>'
    if 'before_x_m' not in r:
        table += '<td colspan="4">No accepted observations; no estimated pose or error.</td></tr>'
        continue
    table += f'<td>{xy(r,"before")}</td><td>{100*r["before_error_m"]:.2f}</td><td>{xy(r,"after")}</td><td>{100*r["after_error_m"]:.4f}</td></tr>'
    raw_table += f'<tr><td>{label}</td><td>{xy(r,"recorded_mean")}</td><td>{r["recorded_mean_error_m"]*100:.2f}</td><td>({r["archived_summary_x_m"]:.3f}, {r["archived_summary_y_m"]:+.3f})</td></tr>'
component_table = ''.join(f'<tr><td>{r["tag_id"]}</td><td>{r["before_error_x_m"]:+.4f}</td><td>{r["before_error_y_m"]:+.4f}</td><td>{r["after_error_x_m"]:+.4f}</td><td>{r["after_error_y_m"]:+.4f}</td></tr>' for r in rows[:3])
sensitivity_table = ''.join(f'<tr><td>{r["odometry_sigma_scale"]}×</td><td>{r["tag0_error_m"]*100:.2f}</td><td>{r["tag2_error_m"]*100:.2f}</td><td>{r["held_out_rmse_m"]*100:.2f}</td></tr>' for r in sensitivity)
png = base64.b64encode((root / 'comparison.png').read_bytes()).decode()
ev = data['independent_evaluation']
iterations = max(h['iteration'] for h in results['history'])
html_text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>AprilTag pose graph results</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#192c40;max-width:1150px;margin:40px auto;padding:0 24px}}h1{{font-size:30px}}h2{{margin-top:36px;font-size:22px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 10px;text-align:right;border-bottom:1px solid #dce2e8}}th{{background:#edf3f6}}th:first-child,td:first-child{{text-align:left}}.note{{background:#f0f5f6;padding:16px;border-left:4px solid #008477}}code{{font-size:13px}}img{{max-width:100%}}a{{color:#007b76}}.small{{font-size:14px;color:#4b5a6b}}</style>
<h1>AprilTag pose graph optimisation</h1><p>26 October 2026 · Position relative to the marked start · Metres · +X forward, +Y right</p>
<p class="note"><strong>Tag 1 is the sole ground-truth anchor at (1.200, −0.700) m.</strong> The existing Python Gauss–Newton mapper converged in {iterations} iterations. Independent horizontal RMSE for tags 0 and 2 decreased from {ev['initial']['horizontal_rmse_m']*100:.2f} cm to {ev['optimized']['horizontal_rmse_m']*100:.2f} cm ({(1-ev['optimized']['horizontal_rmse_m']/ev['initial']['horizontal_rmse_m'])*100:.1f}% reduction). Tag 2 remains tentative.</p>
<img src="data:image/png;base64,{png}" alt="Tag position map and before/after horizontal errors">
<h2>1. Before and after the existing mapper</h2><p>Before = raw-corner ray intersections with the floor, followed by the existing medoid selection, <strong>before injecting any anchor</strong>. After = joint optimisation of all drone SE(3) poses and floor tag X/Y/yaw. Error = √[(X − Xtruth)² + (Y − Ytruth)²]. Tag Z is identically zero by model assumption; supplied ground truth does not allow orientation-error evaluation.</p>
<table><tr><th>Tag</th><th>Sightings</th><th>Ground truth X, Y (m)</th><th>Before X, Y (m)</th><th>Error (cm)</th><th>After X, Y (m)</th><th>Error (cm)</th></tr>{table}</table>
<p class="small">The anchor's very small residual is expected from its imposed prior and is not an independent accuracy measurement. Tag 2 has two accepted frames, below the original confirmation minimum of three. Tag 3 has no accepted observations. The standard mapper exports seed the anchor at truth in their initial fields; comparison.csv and unanchored_initial_tags.json preserve the genuine pre-anchor estimates used above.</p>
<h2>2. Calculation time</h2><table><tr><th>Operation</th><th>Wall time (s)</th></tr>
<tr><td>ZIP loading and input adaptation</td><td>{timing['zip_load_and_adapter_s']:.4f}</td></tr>
<tr><td>Existing approximate synchronisation</td><td>{timing['synchronization_s']:.4f}</td></tr>
<tr><td>Before: graph initialisation only</td><td>{timing['before_initialization_s']:.4f}</td></tr>
<tr><td>Before: total through initialisation</td><td>{timing['baseline_pipeline_s']:.4f}</td></tr>
<tr><td>Anchored graph construction</td><td>{timing['anchored_graph_build_s']:.4f}</td></tr>
<tr><td>After: Gauss–Newton solve, including rank checks</td><td>{timing['optimization_s']:.4f}</td></tr>
<tr><td>After: pipeline through solve, excluding export and separate baseline build</td><td>{timing['optimized_pipeline_excluding_export_s']:.4f}</td></tr>
<tr><td>Analysis driver including baseline build and result export</td><td>{timing['total_driver_including_export_s']:.4f}</td></tr></table>
<p class="small">Measured with time.perf_counter on this computer, one primary run, BLAS/OpenMP threads set to 1. Python startup/imports, report generation and sensitivity reruns are excluded. These are computation times, not recording/flight durations.</p>
<h2>3. Original estimates saved by the recording</h2><p>The archive's detected positions are a separate pre-optimisation baseline. They use the recording's translation estimator, rather than the existing mapper's calibrated raw-corner/floor initialiser. Both are retained so the effect of reinitialisation is visible.</p>
<table><tr><th>Tag</th><th>Mean of all accepted recorded positions X, Y (m)</th><th>Mean-position error (cm)</th><th>Archive summary X, Y (m)</th></tr>{raw_table}</table>
<p class="small">The mean is recomputed from observations.csv; the archive summary differs slightly and is preserved as recorded. The corresponding post-optimisation positions are in the first table. Do not attribute the difference between the recorded estimator and mapper initialiser to optimisation alone.</p>
<h2>4. Signed error components</h2><table><tr><th>Tag</th><th>Before ΔX (m)</th><th>Before ΔY (m)</th><th>After ΔX (m)</th><th>After ΔY (m)</th></tr>{component_table}</table>
<h2>5. Method and assumptions</h2><p>The adapter calls the existing <code>tag_mapping.synchronization.synchronize</code>, <code>graph.build_graph</code>, <code>solver.solve</code> and <code>reporting.evaluate/export_results</code>. The backend is explicit Gauss–Newton with right SE(3) updates, undamped sparse LSMR, block Huber loss and backtracking, following the usual <a href="https://ceres-solver.readthedocs.io/latest/nnls_tutorial.html#pose-graph-2d">relative-pose least-squares formulation</a>. No replacement optimiser was used. Two numerical fixes were needed in the existing solver: a dense rank SVD for moderate graphs and column scaling of the linear unknowns. Neither adds damping or changes the fitted objective.</p>
<ul><li>230 drone pose nodes, 3 tag nodes, 160 pixel observation factors, 229 motion factors, one anchor prior, one start-height prior and one start-attitude prior. All accepted camera frames retain their own pose. Motion support is sampled at 1-second minimum intervals.</li>
<li>Start XY is fixed at (0, 0); initial attitude and height have finite priors. All positions and full attitudes are rotated by −{np.degrees(0.3043428957462311):.4f}° to use the recorded start-forward/right axes. This is an initial-frame conversion, with no alignment to held-out ground truth.</li>
<li>Use the archive's 25° camera pitch, 0.020 m forward lever arm and 0.120 m tag side length. Intrinsics/distortion are the existing repository 320×240 checkerboard calibration. The archive does not provide its own independent calibration identifier.</li>
<li>Floor datum is approximate: assume native PX4 Z=0 is the floor. Start height is the telemetry proxy 0.02463 m, with an assumed 0.050 m prior sigma. The recording has no independent measured floor/body-height datum.</li>
<li>Exposure timing is unverified. The existing legacy replay path uses recorded host-derived epochs. Maximum bracket gap is explicitly 1.1 seconds to cover the 1.0283-second logged gap. Fifteen duplicate host-time telemetry rows are coalesced by retaining the final snapshot at each timestamp. Original inputs are unchanged.</li>
<li>Existing fallback odometry sigmas are 0.10 m translation and [0.05, 0.05, 0.08] rad rotation per 0.1 s, scaled by √(Δt/0.1). Corner sigma is 0.7 px. Anchor XY sigma is assumed 0.01 m. These are heuristic weights, not calibrated uncertainty or independent PX4 samples.</li>
<li>Tags 0, 2 and 3 ground-truth positions are opened for evaluation after fitting. Only tag 1 supplies ground truth to the graph. Rank is verified full; robust cost decreases on all accepted steps. The reported optimisation status is <code>converged_cost</code>.</li></ul>
<h2>6. Sensitivity to odometry weights</h2><p>Halving and doubling the assumed odometry standard deviations both converge. The nominal weights above were kept; no held-out ground-truth values were used to select weights.</p>
<table><tr><th>Odometry sigma multiplier</th><th>Tag 0 error (cm)</th><th>Tag 2 error (cm)</th><th>Independent RMSE (cm)</th></tr>{sensitivity_table}</table>
<h2>7. Reproduce and inspect</h2><p>From the repository root:</p>
<pre><code>MPLCONFIGDIR=/tmp/pose_graph_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\
python laptop/analyze_observations.py /home/maxwell-guico/Downloads/observations.zip \\
  --output outputs/observations_reproduction</code></pre>
<p>Full precision: <a href="comparison.csv">comparison.csv</a>. Graph and trajectories: <a href="validated/results.json">results.json</a>, <a href="validated/trajectory.csv">trajectory.csv</a>, <a href="validated/config.yaml">effective config</a>, <a href="validated/iterations.csv">iteration history</a>. Source hashes are in results.json. The source ZIP was read without modification.</p>
<p class="small">Validation: 23 mapper solver, geometry, replay and end-to-end tests passed. One live-recorder test was excluded because pymavlink is not installed; no live recording was needed for this analysis. The real dataset also passes convergence, full-rank, monotone accepted cost, exact floor Z, fixed start XY and held-out-survey separation checks.</p></html>'''
(root / 'report.html').write_text(html_text)
print(json.dumps({'report': str(root / 'report.html'), 'plot': str(root / 'comparison.png'), 'sensitivity': sensitivity}, indent=2))
