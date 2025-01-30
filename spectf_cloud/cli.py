import rich_click as click
from rich.traceback import install
from trogon import tui
import time, datetime

from typing import Any, Dict, Optional, List

import spectf

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

install()

## TODO: Set up any logging here

help = click.option(
    "-h",
    "--help",
    help="Display help information and then exit.",
    is_flag=True,
    default=None,
)

def get_time() -> tuple[float, str]:
    timez: int = time.perf_counter_ns()
    current_time = time.time()
    timez_stamp: str = datetime.datetime.fromtimestamp(current_time).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    return (timez / 1_000_000_000), timez_stamp


@tui()
@click.group(context_settings=CONTEXT_SETTINGS, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Display the current SpecTf version.",
)
@click.pass_context
def spectf_cloud(
    ctx: click.Context,
    version: bool,
) -> None:
    cli_start, cli_start_ts = get_time()
    ctx.obj.start = cli_start
    ctx.obj.start_ts = cli_start_ts
    if version:
        print(f"SpecTf Cloud version {spectf_cloud.__version__}")
        return


# TODO: Add in summary logging
# @spectf.result_callback()
# @click.pass_context
# def log_after_spectf_get(
#     ctx: click.Context, *args: Optional[List[str]], **kwargs: Optional[Dict[str, Any]]
# ) -> None:
#     end_time, end_time_stamp = get_time()
#     subcmd: str = ctx.invoked_subcommand
#     start_time = ctx.obj.start
#     start_time_stamp = ctx.obj.start_ts
#     cmd_duration = end_time - start_time
#     log.debug(
#         "CLI Summary",
#         start_time_stamp=start_time_stamp,
#         end_time_stamp=end_time_stamp,
#         cmd_duration=cmd_duration,
#         subcmd=subcmd,
#         args=args,
#         kwargs=kwargs,
#     )


class SpecTfCliMetadata(object):
    start: Optional[float]
    start_ts: Optional[str]

## Add in all of the subcommands here
from spectf_cloud import train, deploy, evaluation, comparison_models

def main() -> None:
    spectf_cloud(obj=SpecTfCliMetadata())
