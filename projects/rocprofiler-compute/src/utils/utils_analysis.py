##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

"""
Analysis-specific utilities called ONLY by the analyze code path.
These functions are only called during 'rocprof-compute analyze' operations,
not during profiling.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from vendored import yaml

import config
from utils.kernel_name_shortener import kernel_name_shortener
from utils.logger import console_error, console_log, console_warning, demarcate
from utils.utils_common import NS_TO_MS


@dataclass
class KernelStats:
    """Aggregated kernel launch stats for one kernel name."""

    launches: int = 0
    total_duration_ns: float = 0.0
    kernel_id: Optional[int] = None


@dataclass
class CallTreeNode:
    """A node in the operator call tree.

    Children are ordered operator sub-calls; kernels maps each kernel name to
    a ``KernelStats`` record. ``kernel_launches``
    and ``total_duration_ms`` are inclusive (rolled up from all descendants).
    """

    name: str
    children: dict[str, "CallTreeNode"] = field(default_factory=dict)
    kernels: dict[str, KernelStats] = field(default_factory=dict)
    kernel_launches: int = 0
    total_duration_ms: float = 0.0


def simplify_kernel_name(full_kernel_name: str) -> str:
    """Simplify a kernel name for display by stripping templates and namespaces.

    Intended as a last-resort readability pass *after* the standard
    kernel_name_shortener / process_single_kernel_name pipeline.  Strips
    ``void`` prefix, template parameters, and namespace qualifiers so that
    e.g. ``void at::native::vectorized_elementwise_kernel<4, ...>`` becomes
    ``vectorized_elementwise_kernel``.
    """
    name = full_kernel_name.strip()
    if name.startswith("void "):
        name = name[5:]

    if "<" in name:
        main_part = name.split("<")[0]
    elif "(" in name:
        main_part = name.split("(")[0]
    else:
        main_part = name

    if "::" in main_part:
        function_name = main_part.split("::")[-1].strip()
        return function_name if function_name else name.strip()

    return main_part.strip()


def sanitize_torch_operator_key(name: str) -> str:
    """Normalize a user-supplied operator name to match torch_trace CSV filename stems.

    Strips the ``torch.`` prefix and replaces dots with underscores so that
    inputs like ``torch.nn.functional.conv2d`` become ``nn_functional_conv2d``.
    """
    return name.replace("torch.", "").replace(".", "_")


def get_unique_invocations(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Deduplicate operator invocations from trace rows.

    Each trace row represents one (operator, kernel, counter) combination, so a
    single operator invocation appears in many rows.  Deduplication uses
    (Operator_Name, Context_Id, Start_Timestamp_function) when Context_Id is
    present; otherwise falls back to
    (Operator_Name, Start_Timestamp_function, End_Timestamp_function).

    Returns a DataFrame with at least Operator_Name, Start_Timestamp_function,
    and End_Timestamp_function columns, or None when timestamps are missing.
    """
    has_ts = (
        "Start_Timestamp_function" in df.columns
        and "End_Timestamp_function" in df.columns
    )
    if not has_ts:
        return None

    use_context = "Context_Id" in df.columns and df["Context_Id"].notna().any()
    if use_context:
        return df[
            [
                "Operator_Name",
                "Context_Id",
                "Start_Timestamp_function",
                "End_Timestamp_function",
            ]
        ].drop_duplicates(
            subset=["Operator_Name", "Context_Id", "Start_Timestamp_function"]
        )
    return df[
        ["Operator_Name", "Start_Timestamp_function", "End_Timestamp_function"]
    ].drop_duplicates()


def compute_operator_prefix_stats(
    df: pd.DataFrame,
) -> dict[str, tuple[float, int]]:
    """Compute total duration (ms) and invocation count per operator path prefix.

    Hierarchy semantics: the trace only has the full path per row (e.g. A/B/C).
    We attribute each invocation's duration to every prefix along its path, so
    stats at each node are inclusive.
    Returns dict: prefix -> (total_duration_ms, count).
    """
    invocations = get_unique_invocations(df)
    if invocations is None:
        return {}

    prefix_stats: dict[str, tuple[float, int]] = {}
    for _, row in invocations.iterrows():
        op = str(row["Operator_Name"])
        duration_ns = float(row["End_Timestamp_function"]) - float(
            row["Start_Timestamp_function"]
        )
        duration_ms = duration_ns * NS_TO_MS
        parts = op.split("/")
        for i in range(1, len(parts) + 1):
            prefix = "/".join(parts[:i])
            if prefix not in prefix_stats:
                prefix_stats[prefix] = (0.0, 0)
            prev_dur, prev_cnt = prefix_stats[prefix]
            prefix_stats[prefix] = (prev_dur + duration_ms, prev_cnt + 1)
    return prefix_stats


def build_kernel_name_to_id(
    dfs: list[pd.DataFrame], kernel_verbose: int = 1
) -> Optional[dict[str, int]]:
    """Build a mapping from shortened kernel name to a stable numeric ID.

    Collects every unique Kernel_Name across the supplied DataFrames,
    shortens them with kernel_name_shortener, and assigns sequential IDs
    in alphabetical order.  Returns None when no kernel names are found.
    """
    all_kernel_names: set[str] = set()
    for df in dfs:
        if "Kernel_Name" in df.columns:
            all_kernel_names.update(df["Kernel_Name"].dropna().unique())
    if not all_kernel_names:
        return None
    kernel_df = pd.DataFrame({"Kernel_Name": sorted(all_kernel_names)})
    kernel_df = kernel_name_shortener(kernel_df.copy(), kernel_verbose)
    if kernel_df is None:
        return None
    return {str(row["Kernel_Name"]).strip(): i for i, row in kernel_df.iterrows()}


def is_workload_empty(path: str) -> None:
    """Peek workload directory to verify valid profiling output"""
    workload_dir = Path(path)
    pmc_perf_path = workload_dir / "pmc_perf.csv"

    # Find PMC data files (merged or separate)
    if pmc_perf_path.is_file():
        files_to_check = [pmc_perf_path]
    else:
        pmc_files = list(workload_dir.glob("pmc_perf_*.csv"))
        results_files = list(workload_dir.glob("results_*.csv"))
        files_to_check = pmc_files if pmc_files else results_files

    if not files_to_check:
        console_error("analysis", "No profiling data found.")
        return

    # Validate files are not empty
    for file_path in files_to_check:
        temp_df = pd.read_csv(file_path)
        if temp_df.dropna().empty:
            console_error(
                "profiling",
                f"Found empty cells in {file_path}.\nProfiling data could be corrupt.",
            )
            break


def load_yaml(filepath: str) -> dict[str, Any]:
    """Load YAML file and return as dictionary."""
    with open(filepath) as f:
        return yaml.safe_load(f)


def get_panel_alias() -> dict[str, str]:
    panel_yaml = load_yaml(
        f"{config.rocprof_compute_home}/rocprof_compute_soc/analysis_configs/gfx9_config_template.yaml"
    )
    return {
        panel["panel_alias"]: str(panel["panel_id"]) for panel in panel_yaml["panels"]
    }


def format_scientific_notation_if_needed(
    value: Union[int, float],
    align: str = ">",
    width_align: int = 6,
    precision: int = 2,
    fmt_type_align: str = "f",
    max_length: int = 6,  # Deprecated: kept for backward compatibility
    sci_lower_bound: float = 1e-2,
    sci_upper_bound: float = 1e6,
) -> str:
    """
    Format a numeric value as normal or scientific notation string.

    Uses scientific notation only if it results in a shorter string than
    normal notation, or if the value falls outside the bounds:
    - abs(value) < sci_lower_bound (but not zero)
    - abs(value) >= sci_upper_bound

    Parameters:
    - value: numeric value to format
    - align: alignment character ('<', '>', '^', '=')
    - width_align: total width of formatted output
    - precision: number of digits after decimal point
    - fmt_type_align: format type, e.g., 'f', 'e', 'g'
    - max_length: deprecated, no longer used
    - sci_lower_bound: lower bound for scientific notation usage
    - sci_upper_bound: upper bound for scientific notation usage

    Returns:
    - formatted string according to the criteria, respecting alignment
    """
    del max_length  # Unused, kept for backward compatibility

    abs_val = abs(value)
    use_sci = False

    # Build format specifiers
    normal_format_spec = f"{align}{width_align}.{precision}{fmt_type_align}"
    sci_format_spec = f"{align}{width_align}.{precision}e"

    normal_str = None  # will hold formatted normal string (with padding)
    sci_str = None  # will hold formatted scientific string (with padding)

    if abs_val != 0:
        if abs_val < sci_lower_bound or abs_val >= sci_upper_bound:
            use_sci = True
        else:
            try:
                normal_str = format(value, normal_format_spec)
                normal_str_strip = normal_str.strip()

                sci_str = format(value, sci_format_spec)
                sci_str_strip = sci_str.strip()

                # Decide based on length of stripped strings (ignore padding)
                # Only use scientific notation if it's actually shorter
                if len(sci_str_strip) < len(normal_str_strip):
                    use_sci = True
            except Exception:
                # Fallback to scientific if formatting fails
                use_sci = True

    if use_sci:
        if sci_str is None:
            sci_str = format(value, sci_format_spec)
        formatted = sci_str
    else:
        if normal_str is None:
            normal_str = format(value, normal_format_spec)
        formatted = normal_str

    return formatted


def reverse_multi_index_df_pmc(
    final_df: pd.DataFrame,
) -> tuple[list[pd.DataFrame], list[Any]]:
    """
    Util function to decompose multi-index dataframe.
    """
    # Check if the columns have more than one level
    if not isinstance(final_df.columns, pd.MultiIndex) or final_df.columns.nlevels < 2:
        raise ValueError("Input DataFrame does not have a multi-index column.")

    # Extract the first level of the MultiIndex columns (the file names)
    coll_levels = final_df.columns.get_level_values(0).unique().tolist()

    # Initialize the list of DataFrames
    dfs: list[pd.DataFrame] = []

    # Loop through each 'coll_level' and rebuild the DataFrames
    for level in coll_levels:
        # Select columns that belong to the current 'coll_level'
        columns_for_level = final_df.xs(level, axis=1, level=0)
        # Append the DataFrame for this level
        if isinstance(columns_for_level, pd.Series):
            columns_for_level = columns_for_level.to_frame()
        dfs.append(columns_for_level)

    # Return the list of DataFrames and the column levels
    return dfs, coll_levels


def impute_counters_iteration_multiplex(
    df_multi_index: pd.DataFrame,
    policy: str,
) -> pd.DataFrame:
    """
    Perform data imputation for missing counter values due to iteration multiplexing.
    """
    from utils.logger import console_debug

    non_counter_column_index = [
        "Dispatch_ID",
        "GPU_ID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Kernel_ID",
    ]
    result_dfs: list[pd.DataFrame] = []
    dfs, coll_levels = reverse_multi_index_df_pmc(df_multi_index)

    for df in dfs:
        # Group by unique kernel configurations
        unique_occurences = (
            df.groupby("Kernel_Name")
            if policy == "kernel"
            else df.groupby(
                [
                    "Kernel_Name",
                    "Grid_Size",
                    "Workgroup_Size",
                    "LDS_Per_Workgroup",
                ],
                as_index=False,
            )
        )

        counter_columns = [
            col for col in df.columns if col not in non_counter_column_index
        ]
        # Collect imputed groups as dataframes
        group_dfs = []

        # Log imputation task summary before processing
        console_debug(
            f"Performing data imputation on {len(df)} dispatches "
            f"across {unique_occurences.ngroups} unique kernel configurations"
        )

        for _, group in unique_occurences:
            # Identify counter buckets
            counter_groups: set[frozenset[str]] = set()
            for _, row in group.iterrows():
                # Set of counter column names with non empty values
                cols_frozenset = frozenset(row[counter_columns].dropna().index)
                # If no counters found for this dispatch, continue
                if not cols_frozenset:
                    continue
                # Since counter buckets are repeated in round robin fashion,
                # we can stop once we see a repeated bucket
                if cols_frozenset in counter_groups:
                    break
                counter_groups.add(cols_frozenset)

            # If no counters found for this group, continue
            if not counter_groups:
                continue

            # Iterate over subgroups of dispatches containing
            # all counters and impute missing values
            # Create subgroup_id column for groupby: 0,0,0,...,1,1,1,...,2,2,2,...
            # Use numpy for vectorized operation
            group_copy = group.copy()
            group_copy["__subgroup_id"] = np.arange(len(group_copy)) // len(
                counter_groups
            )
            # groupby().bfill() automatically excludes the grouping column from result
            group_copy[counter_columns] = (
                group_copy[[*counter_columns, "__subgroup_id"]]
                .groupby("__subgroup_id", group_keys=False)
                .bfill()  # Propagate first valid value backward to start of subgroup
                .ffill()  # Propagate forward to end of subgroup
            )
            group_dfs.append(group_copy)

        # Create a new dataframe by concatenating all groups
        result_dfs.append(
            pd.concat(group_dfs, ignore_index=True)
            if group_dfs
            else pd.DataFrame(df.columns)
        )

    final_df = pd.concat(result_dfs, keys=coll_levels, axis=1, copy=False)
    return final_df


def merge_counters_spatial_multiplex(df_multi_index: pd.DataFrame) -> pd.DataFrame:
    """
    For spatial multiplexing, this merges counter values for the same kernel that
    runs on different devices. For time stamp, start time stamp will use median
    while for end time stamp, it will be equal to the summation between median
    start stamp and median delta time.
    """
    non_counter_column_index = [
        "Dispatch_ID",
        "GPU_ID",
        "Queue_ID",
        "PID",
        "TID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Wave_Size",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Correlation_ID",
        "Kernel_ID",
        "Node",
    ]

    expired_column_index = [
        "Node",
        "PID",
        "TID",
        "Queue_ID",
    ]

    result_dfs: list[pd.DataFrame] = []

    # TODO: will need to optimize to avoid this conversion to single index format
    # and do merge directly on multi-index dataframe
    dfs, coll_levels = reverse_multi_index_df_pmc(df_multi_index)

    for df in dfs:
        kernel_name_column_name = "Kernel_Name"
        if "Kernel_Name" not in df and "Name" in df:
            kernel_name_column_name = "Name"

        # Find the values in Kernel_Name that occur more than once
        kernel_single_occurances = df[kernel_name_column_name].value_counts().index

        # Define a list to store the merged rows
        result_data: list[dict[str, Any]] = []

        for kernel_name in kernel_single_occurances:
            # Get all rows for the current kernel_name
            group = df[df[kernel_name_column_name] == kernel_name]

            # Create a dictionary to store the merged row for the current group
            merged_row: dict[str, Any] = {}

            # Process non-counter columns
            for col in [
                col
                for col in non_counter_column_index
                if col not in expired_column_index
            ]:
                if col == "Start_Timestamp":
                    # For Start_Timestamp, take the median
                    merged_row[col] = group["Start_Timestamp"].median()
                elif col == "End_Timestamp":
                    # For End_Timestamp, calculate the median delta time
                    delta_time = group[col] - group["Start_Timestamp"]
                    merged_row[col] = group["Start_Timestamp"] + delta_time.median()
                else:
                    # For other non-counter columns, take the first occurrence (0th row)
                    merged_row[col] = group.iloc[0][col]

            # Process counter columns (assumed to be all columns not in
            # non_counter_column_index)
            counter_columns = [
                col for col in group.columns if col not in non_counter_column_index
            ]
            for counter_col in counter_columns:
                # for counter columns, take the first non-none (or non-nan) value
                current_valid_counter_group = group[group[counter_col].notna()]
                first_valid_value = (
                    current_valid_counter_group.iloc[0][counter_col]
                    if len(current_valid_counter_group) > 0
                    else None
                )
                merged_row[counter_col] = first_valid_value

            # Append the merged row to the result list
            result_data.append(merged_row)

        # Create a new DataFrame from the merged rows
        result_dfs.append(pd.DataFrame(result_data))

    final_df = pd.concat(result_dfs, keys=coll_levels, axis=1, copy=False)
    return final_df


def parse_top_level_location(context_id: object) -> str:
    """Extract 'file:line' from the first entry of a Context_Id string.

    Context_Id format: ``10@main.py:60/#10@main.py:21/...``
    Returns ``main.py:60`` or ``unknown:0`` on failure.
    """
    if pd.isna(context_id) or not str(context_id).strip():
        return "unknown:0"
    first_entry = str(context_id).split("/")[0]
    if "@" not in first_entry:
        return "unknown:0"
    _, location = first_entry.split("@", 1)
    return location if ":" in location else "unknown:0"


def rollup_node_stats(node: CallTreeNode) -> tuple[int, float]:
    """Bottom-up rollup: set kernel_launches and total_duration_ms inclusive."""
    launches = sum(stats.launches for stats in node.kernels.values())
    duration_ns = sum(stats.total_duration_ns for stats in node.kernels.values())

    for child in node.children.values():
        child_launches, child_duration_ns = rollup_node_stats(child)
        launches += child_launches
        duration_ns += child_duration_ns

    node.kernel_launches = launches
    node.total_duration_ms = duration_ns * NS_TO_MS
    return launches, duration_ns


def build_call_trees(
    df: pd.DataFrame,
) -> dict[str, CallTreeNode]:
    """Build per-source-location call trees from a consolidated torch trace DataFrame.

    Returns a dict mapping ``file:line`` to a CallTreeNode root whose
    children form the full operator/kernel hierarchy.

    Each kernel entry is stored as a ``KernelStats`` record.
    Full kernel names are used; shortening is left to the display layer.
    """
    required = {"Operator_Name", "Kernel_Name"}
    if df.empty or not required.issubset(df.columns):
        return {}

    has_kernel_timestamps = (
        "Start_Timestamp_kernel" in df.columns and "End_Timestamp_kernel" in df.columns
    )
    has_context_id = "Context_Id" in df.columns
    has_kernel_id = "Kernel_ID" in df.columns

    deduplication_columns = ["Operator_Name", "Kernel_Name"]
    if has_kernel_timestamps:
        deduplication_columns.append("Start_Timestamp_kernel")
    if has_context_id:
        deduplication_columns.append("Context_Id")
    dispatches = df.drop_duplicates(subset=deduplication_columns)

    call_trees: dict[str, CallTreeNode] = {}

    for row in dispatches.itertuples(index=False):
        op_path = str(row.Operator_Name).strip()
        kernel_name = str(row.Kernel_Name).strip()
        if not op_path or not kernel_name:
            continue

        context_id = getattr(row, "Context_Id", None) if has_context_id else None
        location = parse_top_level_location(context_id)

        duration_ns = 0.0
        if has_kernel_timestamps:
            try:
                duration_ns = float(row.End_Timestamp_kernel) - float(
                    row.Start_Timestamp_kernel
                )
            except (ValueError, TypeError):
                pass

        if location not in call_trees:
            call_trees[location] = CallTreeNode(name=location)
        location_root = call_trees[location]

        current_node = location_root
        for path_segment in op_path.split("/"):
            if path_segment not in current_node.children:
                current_node.children[path_segment] = CallTreeNode(name=path_segment)
            current_node = current_node.children[path_segment]

        if kernel_name not in current_node.kernels:
            kernel_id = None
            kernel_id_value = getattr(row, "Kernel_ID", None) if has_kernel_id else None
            if pd.notna(kernel_id_value):
                kernel_id = int(kernel_id_value)
            current_node.kernels[kernel_name] = KernelStats(kernel_id=kernel_id)
        current_node.kernels[kernel_name].launches += 1
        current_node.kernels[kernel_name].total_duration_ns += duration_ns

    for location_root in call_trees.values():
        rollup_node_stats(location_root)

    return call_trees


def write_torch_trace_consolidated_csv(
    consolidated_df: pd.DataFrame,
    torch_trace_path: Path,
) -> None:
    """Write the consolidated torch trace DataFrame to ``consolidated.csv``."""
    output_file = torch_trace_path / "consolidated.csv"
    consolidated_df.sort_values("Operator_Name", ignore_index=True).to_csv(
        output_file, index=False
    )
    console_log(f"Saved consolidated trace to {output_file}")


def build_call_trees_with_kernel_ids(
    consolidated_df: pd.DataFrame,
    kernel_top_df: pd.DataFrame,
) -> dict[str, CallTreeNode]:
    """Attach Kernel_ID values and build call trees from consolidated trace rows."""
    kernel_name_to_id = {
        str(row["Kernel_Name"]).strip(): idx for idx, row in kernel_top_df.iterrows()
    }
    consolidated_with_ids = consolidated_df.copy()
    consolidated_with_ids["Kernel_ID"] = (
        consolidated_with_ids["Kernel_Name"].str.strip().map(kernel_name_to_id)
    )
    return build_call_trees(consolidated_with_ids)


@demarcate
def process_torch_trace_output(
    workload_dir: str,
) -> tuple[pd.DataFrame, Path]:
    """
    Build consolidated torch trace rows and prepare output directory.

    - Performs inner join on Correlation_ID, filtering out unmatched entries
    - Consolidates data across passes and normalizes required columns
    - Prepares a clean workload_dir/torch_trace/ directory for output files

    Returns (consolidated_df, torch_trace_path) on success.
    """
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    marker_api_trace_csvs = list(
        Path(workload_dir).glob("**/torch_trace*_marker_api_trace.csv")
    )
    counter_collection_csvs = [
        markers_file.parent
        / markers_file.name.replace("_marker_api_trace.", "_counter_collection.")
        for markers_file in marker_api_trace_csvs
    ]
    existing_csv_files = [
        [marker_api_trace_csvs[i], counter_collection_csvs[i]]
        for i in range(len(marker_api_trace_csvs))
        if counter_collection_csvs[i].is_file() and marker_api_trace_csvs[i].is_file()
    ]

    if not existing_csv_files:
        console_warning(
            "No marker files with corresponding counter files found. "
            "Ensure profiling was done with '--torch-trace'."
        )
        return pd.DataFrame(), Path(f"{workload_dir}/torch_trace")

    torch_trace_path = Path(f"{workload_dir}/torch_trace")
    if torch_trace_path.exists():
        shutil.rmtree(torch_trace_path)
        console_log(f"Removed previous torch_trace directory: {torch_trace_path}")
    torch_trace_path.mkdir(parents=True, exist_ok=True)

    # Join marker and counter data
    def _merge_pair(
        marker_path: Path,
        counter_path: Path,
        join_keys: tuple[str, ...] = ("Correlation_ID",),
    ) -> pd.DataFrame:
        """Merge a pair of marker and counter csv files on specified keys,
        return the merged dataframe.
        """
        marker_df = pd.read_csv(marker_path)
        counter_df = pd.read_csv(counter_path)
        # Normalize column names to handle case inconsistencies
        marker_df.columns = marker_df.columns.str.replace(
            "Correlation_Id", "Correlation_ID"
        )
        counter_df.columns = counter_df.columns.str.replace(
            "Correlation_Id", "Correlation_ID"
        )

        return pd.merge(
            marker_df,
            counter_df,
            on=join_keys,
            how="inner",
            suffixes=("_function", "_kernel"),
        )

    # If rocpd format, pairs are present in workload_dir, one pair per fbase
    # If csv format, pairs are present in workload/{fbase}/ one pair per process
    # Extracting the output_format used in profiling from the path of a marker file
    if Path(workload_dir).resolve() == existing_csv_files[0][0].parent.resolve():
        join_keys = ("Correlation_ID", "GUID")  # output_format "rocpd"
    else:
        join_keys = ("Correlation_ID",)  # output_format "csv"
    consolidated_df = pd.concat(
        [_merge_pair(f[0], f[1], join_keys) for f in existing_csv_files],
        ignore_index=True,
    )
    required_columns = [
        "Function",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp_function",
        "End_Timestamp_function",
        "Start_Timestamp_kernel",
        "End_Timestamp_kernel",
    ]
    missing_columns = [
        col for col in required_columns if col not in consolidated_df.columns
    ]
    if missing_columns:
        console_error(
            f"Consolidated torch trace is missing required columns {missing_columns}"
        )
        raise ValueError(
            f"Consolidated torch trace is missing required columns {missing_columns}"
        )
    consolidated_df = consolidated_df[required_columns]
    if consolidated_df.isnull().values.any():
        console_warning("Consolidated torch trace contains missing values")
        raise ValueError("Consolidated torch trace contains missing values")
    consolidated_df = consolidated_df.sort_values(by=["Function", "Counter_Name"])
    split_columns = consolidated_df["Function"].str.split(":#", expand=True)
    consolidated_df["Operator_Name"] = (
        split_columns[0] if len(split_columns.columns) > 0 else None
    )
    consolidated_df["Context_Id"] = (
        split_columns[1] if len(split_columns.columns) > 1 else None
    )
    consolidated_df.drop(columns=["Function"], inplace=True)
    consolidated_df = consolidated_df[
        [
            "Operator_Name",
            "Context_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp_function",
            "End_Timestamp_function",
            "Start_Timestamp_kernel",
            "End_Timestamp_kernel",
        ]
    ]
    if consolidated_df.isnull().values.any():
        console_error(
            "Missing values in consolidated torch trace after splitting ",
            "the Function name.",
        )
        raise ValueError("Missing values in consolidated torch trace after splitting")

    return consolidated_df, torch_trace_path
