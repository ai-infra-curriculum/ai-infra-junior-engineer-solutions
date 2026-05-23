#!/usr/bin/env python3
"""
System Monitor CLI Tool

A command-line tool for monitoring system resources including CPU, memory,
disk usage, and running processes.

Usage:
    python system_monitor.py --all
    python system_monitor.py --cpu --memory
    python system_monitor.py --all --output metrics.json
"""

import psutil
import click
import json
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn
from typing import Dict, List


console = Console()


def get_cpu_info() -> Dict:
    """
    Get CPU usage information.

    Returns:
        Dictionary containing CPU metrics
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    return {
        "cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "cpu_frequency_mhz": cpu_freq.current if cpu_freq else None,
        "timestamp": datetime.now().isoformat()
    }


def get_memory_info() -> Dict:
    """
    Get memory usage information.

    Returns:
        Dictionary containing memory metrics
    """
    memory = psutil.virtual_memory()

    return {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_gb": round(memory.used / (1024**3), 2),
        "percent": memory.percent,
        "timestamp": datetime.now().isoformat()
    }


def get_disk_info() -> Dict:
    """
    Get disk usage information.

    Returns:
        Dictionary containing disk metrics
    """
    disk = psutil.disk_usage('/')

    return {
        "total_gb": round(disk.total / (1024**3), 2),
        "used_gb": round(disk.used / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "percent": disk.percent,
        "timestamp": datetime.now().isoformat()
    }


def get_top_processes(n: int = 5) -> List[Dict]:
    """
    Get top N processes by CPU usage.

    Args:
        n: Number of processes to return

    Returns:
        List of process dictionaries
    """
    processes = []

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            proc_info = proc.info
            processes.append({
                "pid": proc_info['pid'],
                "name": proc_info['name'],
                "cpu_percent": proc_info['cpu_percent'],
                "memory_percent": round(proc_info['memory_percent'], 2)
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Sort by CPU usage
    processes.sort(key=lambda x: x['cpu_percent'] or 0, reverse=True)

    return processes[:n]


def display_cpu_info(cpu_info: Dict):
    """Display CPU information in a formatted table."""
    table = Table(title="CPU Information")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("CPU Usage", f"{cpu_info['cpu_percent']}%")
    table.add_row("CPU Count", str(cpu_info['cpu_count']))
    if cpu_info['cpu_frequency_mhz']:
        table.add_row("Frequency", f"{cpu_info['cpu_frequency_mhz']:.0f} MHz")

    console.print(table)


def display_memory_info(memory_info: Dict):
    """Display memory information in a formatted table."""
    table = Table(title="Memory Information")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total", f"{memory_info['total_gb']} GB")
    table.add_row("Used", f"{memory_info['used_gb']} GB")
    table.add_row("Available", f"{memory_info['available_gb']} GB")
    table.add_row("Usage", f"{memory_info['percent']}%")

    console.print(table)


def display_disk_info(disk_info: Dict):
    """Display disk information in a formatted table."""
    table = Table(title="Disk Information")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total", f"{disk_info['total_gb']} GB")
    table.add_row("Used", f"{disk_info['used_gb']} GB")
    table.add_row("Free", f"{disk_info['free_gb']} GB")
    table.add_row("Usage", f"{disk_info['percent']}%")

    console.print(table)


def display_processes(processes: List[Dict]):
    """Display top processes in a formatted table."""
    table = Table(title="Top Processes by CPU")

    table.add_column("PID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("CPU %", style="yellow")
    table.add_column("Memory %", style="magenta")

    for proc in processes:
        table.add_row(
            str(proc['pid']),
            proc['name'],
            f"{proc['cpu_percent'] or 0:.1f}",
            f"{proc['memory_percent']:.2f}"
        )

    console.print(table)


def save_metrics(metrics: Dict, output_path: str):
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[green]✓ Metrics saved to {output_path}[/green]")


@click.command()
@click.option('--cpu', is_flag=True, help='Show CPU information')
@click.option('--memory', is_flag=True, help='Show memory information')
@click.option('--disk', is_flag=True, help='Show disk information')
@click.option('--processes', is_flag=True, help='Show top processes')
@click.option('--all', 'show_all', is_flag=True, help='Show all information')
@click.option('--output', '-o', help='Save metrics to file (JSON)')
@click.option('--watch', '-w', type=int, help='Watch mode - refresh interval in seconds')
def main(cpu, memory, disk, processes, show_all, output, watch):
    """
    System Monitor CLI Tool

    Monitor system resources from the command line.

    Examples:
        python system_monitor.py --all
        python system_monitor.py --cpu --memory
        python system_monitor.py --all --output metrics.json
        python system_monitor.py --all --watch 2
    """
    # If nothing specified, show all
    if not any([cpu, memory, disk, processes, show_all]):
        show_all = True

    # Set flags if --all is specified
    if show_all:
        cpu = memory = disk = processes = True

    def display_metrics():
        """Display all requested metrics."""
        metrics = {}

        console.clear()
        console.print("[bold blue]System Monitor[/bold blue]\n")

        if cpu:
            cpu_info = get_cpu_info()
            display_cpu_info(cpu_info)
            metrics['cpu'] = cpu_info
            console.print()

        if memory:
            memory_info = get_memory_info()
            display_memory_info(memory_info)
            metrics['memory'] = memory_info
            console.print()

        if disk:
            disk_info = get_disk_info()
            display_disk_info(disk_info)
            metrics['disk'] = disk_info
            console.print()

        if processes:
            top_processes = get_top_processes()
            display_processes(top_processes)
            metrics['processes'] = top_processes
            console.print()

        if output and not watch:
            save_metrics(metrics, output)

        return metrics

    # Watch mode
    if watch:
        console.print(f"[yellow]Watch mode: Refreshing every {watch} seconds. Press Ctrl+C to stop.[/yellow]\n")
        try:
            while True:
                display_metrics()
                time.sleep(watch)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
    else:
        # Single run
        display_metrics()


if __name__ == '__main__':
    main()
