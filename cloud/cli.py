import rich_click as click
from rich.traceback import install
from trogon import tui
import time, datetime
import __init__

from typing import Any, Dict, Optional, List

ROOT_COMMAND = "spectf"
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

common_commands = [
    "train",
    "eval",
    "deploy",
]

option_dictionary = click.rich_click.OPTION_GROUPS = {}

# Insert commond options in the dictionary
for command in common_commands:
    option_dictionary.update(
        {
            f"{ROOT_COMMAND} {command}": [
                {
                    "name": "Train",
                    "options": ["--A", "--B"], # TODO: Flag options
                },
                {
                    "name": "Architecture",
                    "options": ["--A", "--B",], # TODO: Flag options
                },
            ]
        }
    )


def get_time() -> tuple[float, str]:
    timez: int = time.perf_counter_ns()
    current_time = time.time()
    timez_stamp: str = datetime.datetime.fromtimestamp(current_time).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    return (timez / 1_000_000_000), timez_stamp


@tui()
@click.group(context_settings=CONTEXT_SETTINGS, chain=True, invoke_without_command=True)
@click.option(
    "--version",
    is_flag=True,
    default=False,
    help="Display the current SpecTf version.",
)
@click.pass_context
def spectf(
    ctx: click.Context,
    version: bool,
) -> None:
    cli_start, cli_start_ts = get_time()
    ctx.obj.start = cli_start
    ctx.obj.start_ts = cli_start_ts
    if version:
        print(f"Spectf version {__init__.__version__}")
        return

    # spectf: SpectfCli = SpectfCli()
    # ctx.obj.spectf = spectf


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


# class SpecTfCliOptions(object):
#     def __init__(self, *args: Any, **kwargs: Optional[Dict[str, Any]]) -> None:
#         [self.__setattr__(kk, vv) for kk, vv in kwargs.items()]

## Add in all of the subcommands here
import train_spectf_cloud #, reference_models, evaluation, deploy

def main() -> None:
    spectf(
        # obj=SpecTfCliOptions(),
    )
