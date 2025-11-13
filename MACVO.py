import argparse
import json
import threading
import time
import torch
import rerun as rr
import numpy as np
import pypose as pp
from pathlib import Path
from contextlib import contextmanager
from typing import Dict

from DataLoader import SequenceBase, StereoFrame, smart_transform
from Evaluation.EvalSeq import EvaluateSequences
from Odometry.MACVO import MACVO

from Utility.Config import load_config, asNamespace
from Utility.PrettyPrint import print_as_table, ColoredTqdm, Logger
from Utility.Sandbox import Sandbox
from Utility.Visualize import fig_plt, rr_plt
from Utility.Timer import Timer


# ============================================================================
# Performance Profiler
# ============================================================================
class Profiler:
    """Thread-safe lightweight profiler for aggregating step runtimes."""

    def __init__(self):
        self._stats = {}
        self._lock = threading.Lock()

    def _update(self, name: str, duration: float):
        with self._lock:
            stat = self._stats.get(name)
            if stat is None:
                stat = {
                    'total_seconds': 0.0,
                    'count': 0,
                    'min_seconds': float('inf'),
                    'max_seconds': 0.0
                }
                self._stats[name] = stat
            stat['total_seconds'] += duration
            stat['count'] += 1
            if duration < stat['min_seconds']:
                stat['min_seconds'] = duration
            if duration > stat['max_seconds']:
                stat['max_seconds'] = duration

    @contextmanager
    def time_block(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._update(name, end - start)

    def record(self, name: str, duration_seconds: float):
        self._update(name, duration_seconds)

    def export(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            result = {}
            for name, stat in self._stats.items():
                avg = stat['total_seconds'] / stat['count'] if stat['count'] else 0.0
                result[name] = {
                    'total_seconds': stat['total_seconds'],
                    'count': stat['count'],
                    'avg_seconds': avg,
                    'min_seconds': (0.0 if stat['min_seconds'] == float('inf') else stat['min_seconds']),
                    'max_seconds': stat['max_seconds']
                }
            return result

# Global profiler instance
profiler = Profiler()


def VisualizeRerunCallback(frame: StereoFrame, system: MACVO, pb: ColoredTqdm):
    with profiler.time_block('visualization.rerun_setup'):
        rr.set_time_sequence("frame_idx", frame.frame_idx)
    
    # Non-key frame does not need visualization
    if system.graph.frames.data["need_interp"][-1]: return
    
    with profiler.time_block('visualization.log_trajectory'):
        if frame.frame_idx > 0:    
            rr_plt.log_trajectory("/world/est", pp.SE3(system.graph.frames.data["pose"].tensor))
    
    with profiler.time_block('visualization.log_camera'):
        rr_plt.log_camera("/world/macvo/cam_left", pp.SE3(system.graph.frames.data["pose"][-1]), system.graph.frames.data["K"][-1])
        rr_plt.log_image ("/world/macvo/cam_left", frame.stereo.imageL[0].permute(1, 2, 0))
    
    with profiler.time_block('visualization.log_map_points'):
        map_points = system.graph.get_frame2map(system.graph.frames[-1:])
        rr_plt.log_points("/world/point_cloud", map_points.data["pos_Tw"], map_points.data["color"], map_points.data["cov_Tw"], "sphere")
    
    with profiler.time_block('visualization.log_vo_points'):
        vo_points  = system.graph.get_match2point(system.graph.get_frame2match(system.graph.frames[-1:]))
        rr_plt.log_points("/world/vo_tracking", vo_points.data["pos_Tw"], vo_points.data["color"], vo_points.data["cov_Tw"], "sphere")
    

def VisualizeVRAMUsage(frame: StereoFrame, system: MACVO, pb: ColoredTqdm):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_reserved(0) / 1e9  # Convert to GB
        allocated_memory = f"{round(allocated_memory, 3)} GB"
    else:
        allocated_memory = "N/A"
    
    pb.set_description(desc=f"{system.graph}, VRAM={allocated_memory}")


def print_performance_analysis(runtime: Dict, total_elapsed: float):
    """Print detailed performance analysis with categorization."""
    if not runtime:
        return
    
    # Define aggregate timers that shouldn't be summed
    aggregate_timers = set()
    
    # Filter to get only leaf operations for accurate percentages
    leaf_operations = {k: v for k, v in runtime.items() if k not in aggregate_timers}
    
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Total elapsed time: {total_elapsed:.3f}s\n")
    
    # 1. SETUP
    setup_ops = {k: v for k, v in leaf_operations.items() if k.startswith('setup.')}
    if setup_ops:
        setup_total = sum(v['total_seconds'] for v in setup_ops.values())
        print(f"{'SETUP':^80}")
        print("-" * 80)
        print(f"Total: {setup_total:.3f}s ({setup_total/total_elapsed*100:.1f}%)\n")
        for name, stats in sorted(setup_ops.items(), key=lambda x: x[1]['total_seconds'], reverse=True):
            print(
                f"  {name:30s} | {stats['total_seconds']:8.3f}s | "
                f"Avg: {stats['avg_seconds']*1000:7.2f}ms"
            )
        print("")
    
    # 2. DATA LOADING
    data_ops = {k: v for k, v in leaf_operations.items() if k.startswith('data.')}
    if data_ops:
        data_total = sum(v['total_seconds'] for v in data_ops.values())
        print(f"{'DATA LOADING':^80}")
        print("-" * 80)
        print(f"Total: {data_total:.3f}s ({data_total/total_elapsed*100:.1f}%)\n")
        for name, stats in sorted(data_ops.items(), key=lambda x: x[1]['total_seconds'], reverse=True):
            pct = (stats['total_seconds'] / data_total * 100) if data_total > 0 else 0
            print(
                f"  {name:30s} | {stats['total_seconds']:8.3f}s ({pct:5.1f}%) | "
                f"Count: {stats['count']:5d} | Avg: {stats['avg_seconds']*1000:7.2f}ms | "
                f"Min: {stats['min_seconds']*1000:6.2f}ms | Max: {stats['max_seconds']*1000:7.2f}ms"
            )
        print("")
    
    # 3. PER-FRAME PROCESSING
    frame_ops = {k: v for k, v in leaf_operations.items() if k.startswith('per_frame.')}
    if frame_ops:
        frame_total = sum(v['total_seconds'] for v in frame_ops.values())
        print(f"{'PER-FRAME PROCESSING':^80}")
        print("-" * 80)
        print(f"Total: {frame_total:.3f}s ({frame_total/total_elapsed*100:.1f}%)\n")
        for name, stats in sorted(frame_ops.items(), key=lambda x: x[1]['total_seconds'], reverse=True):
            pct = (stats['total_seconds'] / frame_total * 100) if frame_total > 0 else 0
            print(
                f"  {name:30s} | {stats['total_seconds']:8.3f}s ({pct:5.1f}%) | "
                f"Count: {stats['count']:5d} | Avg: {stats['avg_seconds']*1000:7.2f}ms | "
                f"Min: {stats['min_seconds']*1000:6.2f}ms | Max: {stats['max_seconds']*1000:7.2f}ms"
            )
        print("")
    
    # 4. VISUALIZATION
    vis_ops = {k: v for k, v in leaf_operations.items() if k.startswith('visualization.')}
    if vis_ops:
        vis_total = sum(v['total_seconds'] for v in vis_ops.values())
        print(f"{'VISUALIZATION':^80}")
        print("-" * 80)
        print(f"Total: {vis_total:.3f}s ({vis_total/total_elapsed*100:.1f}%)\n")
        for name, stats in sorted(vis_ops.items(), key=lambda x: x[1]['total_seconds'], reverse=True):
            pct = (stats['total_seconds'] / vis_total * 100) if vis_total > 0 else 0
            print(
                f"  {name:30s} | {stats['total_seconds']:8.3f}s ({pct:5.1f}%) | "
                f"Count: {stats['count']:5d} | Avg: {stats['avg_seconds']*1000:7.2f}ms | "
                f"Min: {stats['min_seconds']*1000:6.2f}ms | Max: {stats['max_seconds']*1000:7.2f}ms"
            )
        print("")
    
    # 5. I/O OPERATIONS
    io_ops = {k: v for k, v in leaf_operations.items() if k.startswith('io.')}
    if io_ops:
        io_total = sum(v['total_seconds'] for v in io_ops.values())
        print(f"{'I/O OPERATIONS':^80}")
        print("-" * 80)
        print(f"Total: {io_total:.3f}s ({io_total/total_elapsed*100:.1f}%)\n")
        for name, stats in sorted(io_ops.items(), key=lambda x: x[1]['total_seconds'], reverse=True):
            pct = (stats['total_seconds'] / io_total * 100) if io_total > 0 else 0
            print(
                f"  {name:30s} | {stats['total_seconds']:8.3f}s ({pct:5.1f}%) | "
                f"Count: {stats['count']:5d} | Avg: {stats['avg_seconds']*1000:7.2f}ms | "
                f"Min: {stats['min_seconds']*1000:6.2f}ms | Max: {stats['max_seconds']*1000:7.2f}ms"
            )
        print("")
    
    # 6. SUMMARY
    print(f"{'TIME BREAKDOWN SUMMARY':^80}")
    print("=" * 80)
    print("Note: Categories may exceed 100% due to parallel processing\n")
    
    categories_summary = [
        ("Setup", sum(v['total_seconds'] for v in setup_ops.values()) if setup_ops else 0),
        ("Data Loading", sum(v['total_seconds'] for v in data_ops.values()) if data_ops else 0),
        ("Per-Frame Processing", sum(v['total_seconds'] for v in frame_ops.values()) if frame_ops else 0),
        ("Visualization", sum(v['total_seconds'] for v in vis_ops.values()) if vis_ops else 0),
        ("I/O", sum(v['total_seconds'] for v in io_ops.values()) if io_ops else 0),
    ]
    
    for cat_name, cat_time in sorted(categories_summary, key=lambda x: x[1], reverse=True):
        if cat_time > 0:
            pct = (cat_time / total_elapsed * 100) if total_elapsed > 0 else 0
            bar_length = min(int(pct / 2), 50)  # Cap at 50 chars
            bar = "█" * bar_length
            print(f"  {cat_name:25s} | {cat_time:8.3f}s | {pct:6.1f}% | {bar}")
    
    print("-" * 80)
    print(f"  {'Wall-clock time':25s} | {total_elapsed:8.3f}s | 100.0% |\n")
    
    # 7. TOP BOTTLENECKS
    print(f"{'TOP 10 BOTTLENECKS (Leaf Operations Only)':^80}")
    print("-" * 80)
    top_items = sorted(leaf_operations.items(), key=lambda x: x[1]['total_seconds'], reverse=True)[:10]
    for i, (name, stats) in enumerate(top_items, 1):
        pct = (stats['total_seconds'] / total_elapsed * 100) if total_elapsed > 0 else 0
        print(
            f"  {i:2d}. {name:28s} | {stats['total_seconds']:8.3f}s ({pct:5.1f}%) | "
            f"Avg: {stats['avg_seconds']*1000:7.2f}ms | Count: {stats['count']:5d}"
        )
    
    print("\n" + "="*80)
    
    # Print optimization recommendations
    print(f"{'OPTIMIZATION RECOMMENDATIONS':^80}")
    print("=" * 80)
    
    recommendations = []
    
    # Analyze data loading
    data_ops = {k: v for k, v in leaf_operations.items() if k.startswith('data.')}
    if data_ops:
        data_total = sum(v['total_seconds'] for v in data_ops.values())
        if data_total > total_elapsed * 0.15:  # >15% on data
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Data Loading',
                'issue': f'Data operations take {data_total:.1f}s ({data_total/total_elapsed*100:.1f}%)',
                'suggestions': [
                    'Use --preload flag to load entire sequence into RAM',
                    'Reduce image resolution with --downscale flag',
                    'Use faster storage (SSD vs HDD)',
                    'Consider memory-mapped files for large sequences'
                ]
            })
    
    # Analyze odometry processing
    odom_op = runtime.get('per_frame.odometry')
    if odom_op:
        odom_total = odom_op['total_seconds']
        if odom_total > total_elapsed * 0.70:  # >70% on odometry
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Odometry Processing',
                'issue': f'Odometry takes {odom_total:.1f}s ({odom_total/total_elapsed*100:.1f}%)',
                'suggestions': [
                    f'Avg per frame: {odom_op["avg_seconds"]*1000:.1f}ms',
                    'This is the core algorithm - consider:',
                    '  - Reducing feature extraction density',
                    '  - Using faster feature detector/descriptor',
                    '  - Optimizing bundle adjustment iterations',
                    '  - Using GPU acceleration where available'
                ]
            })
    
    # Analyze visualization overhead
    vis_ops = {k: v for k, v in leaf_operations.items() if k.startswith('visualization.')}
    if vis_ops:
        vis_total = sum(v['total_seconds'] for v in vis_ops.values())
        if vis_total > total_elapsed * 0.10:  # >10% on visualization
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Visualization',
                'issue': f'Visualization takes {vis_total:.1f}s ({vis_total/total_elapsed*100:.1f}%)',
                'suggestions': [
                    'Disable rerun visualization with --useRR flag for production',
                    'Reduce point cloud density for visualization',
                    'Visualize every Nth frame instead of all frames',
                    f'Current cost: ~{vis_total/odom_op["count"] if odom_op else 0:.1f}s per frame'
                ]
            })
    
    # Analyze I/O overhead
    io_ops = {k: v for k, v in leaf_operations.items() if k.startswith('io.')}
    if io_ops:
        io_total = sum(v['total_seconds'] for v in io_ops.values())
        if io_total > total_elapsed * 0.05:  # >5% on I/O
            recommendations.append({
                'priority': 'LOW',
                'category': 'I/O Operations',
                'issue': f'I/O operations take {io_total:.1f}s ({io_total/total_elapsed*100:.1f}%)',
                'suggestions': [
                    'I/O overhead is acceptable',
                    'Could batch writes or use async I/O if needed',
                    'Using compressed formats could reduce disk I/O'
                ]
            })
    
    # Check for GPU utilization
    setup_time = sum(v['total_seconds'] for k, v in leaf_operations.items() if k.startswith('setup.'))
    if setup_time > 5.0:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Initialization',
            'issue': f'Setup takes {setup_time:.1f}s',
            'suggestions': [
                'Long initialization time detected',
                'Consider caching loaded models',
                'Pre-compile CUDA kernels if applicable',
                'This is one-time cost, less critical for long sequences'
            ]
        })
    
    # Print recommendations
    if recommendations:
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        for i, rec in enumerate(recommendations, 1):
            priority_color = '🔴' if rec['priority'] == 'HIGH' else '🟡' if rec['priority'] == 'MEDIUM' else '🟢'
            print(f"\n{i}. [{priority_color} {rec['priority']}] {rec['category']}")
            print(f"   Issue: {rec['issue']}")
            print(f"   Suggestions:")
            for j, suggestion in enumerate(rec['suggestions'], 1):
                print(f"     {j}) {suggestion}")
    else:
        print("No major bottlenecks detected. Performance looks good!")
    
    print("\n" + "="*80)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odom", type=str, default = "Config/Experiment/MACVO/MACVO.yaml")
    parser.add_argument("--data", type=str, default = "Config/Sequence/TartanAir_abandonfac_001.yaml")
    parser.add_argument(
        "--seq_to",
        type=int,
        default=None,
        help="Crop sequence to frame# when ran. Set to -1 (default) if wish to run whole sequence",
    )
    parser.add_argument(
        "--seq_from",
        type=int,
        default=0,
        help="Crop sequence from frame# when ran. Set to 0 (default) if wish to start from first frame",
    )
    parser.add_argument(
        "--resultRoot",
        type=str,
        default="./Results",
        help="Directory to store trajectory and files generated by the script."
    )
    parser.add_argument(
        "--useRR",
        action="store_true",
        help="Activate RerunVisualizer to generate <config.Project>.rrd file for visualization.",
    )
    parser.add_argument(
        "--saveplt",
        action="store_true",
        help="Activate PLTVisualizer to generate <frame_idx>.jpg file in space folder for covariance visualization.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload entire trajectory into RAM to reduce data fetching overhead during runtime."
    )
    parser.add_argument(
        "--autoremove",
        action="store_true",
        help="Cleanup result sandbox after script finishs / crashed. Helpful during testing & debugging."
    )
    # parser.add_argument(
    #     "--noeval", 
    #     action="store_true",
    #     help="Evaluate sequence after running odometry."
    # )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Record timing for system (active Utility.Timer for global time recording)"
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Downscale factor for images (e.g., 0.5 for half size, 2.0 for double size). Camera parameters from yaml should match this scale."
    )
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.perf_counter()
    args = get_args()

    # Metadata setup & visualizer setup
    with profiler.time_block('setup.load_configs'):
        cfg, cfg_dict = load_config(Path(args.odom))
        odomcfg, odomcfg_dict = cfg.Odometry, cfg_dict["Odometry"]
        datacfg, datacfg_dict = load_config(Path(args.data))
        project_name = odomcfg.name + "@" + datacfg.name

    with profiler.time_block('setup.create_sandbox'):
        exp_space = Sandbox.create(Path(args.resultRoot), project_name)
        if args.autoremove: exp_space.set_autoremove()
        exp_space.config = {
            "Project": project_name,
            "Odometry": odomcfg_dict,
            "Data": {"args": datacfg_dict, "end_idx": args.seq_to, "start_idx": args.seq_from, "downscale": args.downscale},
        }

    # Setup logging and visualization
    with profiler.time_block('setup.visualization'):
        if args.useRR:
            rr_plt.default_mode = "rerun"
            rr_plt.init_connect(project_name)
        
        Timer.setup(active=args.timing)
        fig_plt.default_mode = "image" if args.saveplt else "none"

    def onFrameFinished(frame: StereoFrame, system: MACVO, pb: ColoredTqdm):
        with profiler.time_block('per_frame.callback'):
            VisualizeRerunCallback(frame, system, pb)
            VisualizeVRAMUsage(frame, system, pb)

    # Initialize data source
    with profiler.time_block('data.instantiate_sequence'):
        base_sequence = SequenceBase[StereoFrame].instantiate(datacfg.type, datacfg.args).clip(args.seq_from, args.seq_to)

    # Apply downscaling if specified
    if args.downscale != 1.0:
        with profiler.time_block('data.downscale'):
            from DataLoader.Transform import ScaleFrame
            from types import SimpleNamespace

            Logger.write("info", f"Applying downscale factor: {args.downscale}")

            # Get original size from first frame
            test_frame = base_sequence[0]
            orig_h, orig_w = test_frame.stereo.height, test_frame.stereo.width
            orig_K = test_frame.stereo.K.clone()

            Logger.write("info", f"Original image size: {orig_w}x{orig_h}")
            Logger.write("info", f"Original K matrix:\n{orig_K[0]}")

            downscale_transform = ScaleFrame(SimpleNamespace(
                scale_u=args.downscale,
                scale_v=args.downscale,
                interp="bilinear"
            ))
            base_sequence = base_sequence.transform(downscale_transform)

            # Verify resize happened
            test_frame_scaled = base_sequence[0]
            new_h, new_w = test_frame_scaled.stereo.height, test_frame_scaled.stereo.width
            new_K = test_frame_scaled.stereo.K

            Logger.write("info", f"Downscaled image size: {new_w}x{new_h}")
            Logger.write("info", f"Downscaled K matrix:\n{new_K[0]}")
            Logger.write("info", f"Memory reduction: {orig_h*orig_w} -> {new_h*new_w} pixels ({100*(1-new_h*new_w/(orig_h*orig_w)):.1f}% reduction)")

    # Apply preprocessing transformations from config
    with profiler.time_block('data.preprocess'):
        sequence = smart_transform(base_sequence, cfg.Preprocess)
    
    with profiler.time_block('data.preload'):
        if args.preload:
            sequence = sequence.preload()
    
    with profiler.time_block('setup.create_system'):
        system = MACVO[StereoFrame].from_config(asNamespace(exp_space.config))
    
    with profiler.time_block('per_frame.odometry'):
        system.receive_frames(sequence, exp_space, on_frame_finished=onFrameFinished)
    
    with profiler.time_block('io.save_results'):
        rr_plt.log_trajectory("/world/est"  , torch.tensor(np.load(exp_space.path("poses.npy"))[:, 1:]))
        try:
            rr_plt.log_points    ("/world/point_cloud", 
                                    system.get_map().map_points.data["pos_Tw"].tensor,
                                    system.get_map().map_points.data["color"].tensor,
                                    system.get_map().map_points.data["cov_Tw"].tensor,
                                    "color")
        except RuntimeError:
            Logger.write("warn", "Unable to log full pointcloud - is mapping mode on?")
    
    Timer.report()
    Timer.save_elapsed(exp_space.path("elapsed_time.json"))
    
    # Print and save performance analysis
    end_time = time.perf_counter()
    total_elapsed = end_time - start_time
    print(f"\nTotal time: {total_elapsed:.3f} seconds")
    
    runtime = profiler.export()
    print_performance_analysis(runtime, total_elapsed)
    
    # Save detailed performance data to JSON
    performance_summary = {
        'total_elapsed_seconds': total_elapsed,
        'project_name': project_name,
        'odom_config': str(args.odom),
        'data_config': str(args.data),
        'result_root': str(args.resultRoot),
        'seq_from': args.seq_from,
        'seq_to': args.seq_to,
        'downscale': args.downscale,
        'preload': args.preload,
        'runtime_breakdown': runtime
    }
    
    performance_path = exp_space.path("performance_summary.json")
    with open(performance_path, 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    print(f"\nPerformance summary saved to: {performance_path}")
