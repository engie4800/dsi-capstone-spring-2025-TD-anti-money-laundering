import io
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph
from torch import Tensor

from pipeline import ModelPipeline
from pipeline.checks import Checker


def estimate_df_to_csv_size(df: pd.DataFrame) -> int:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    size_in_bytes = buffer.tell()
    buffer.close()
    return size_in_bytes


def validate_subgraph_inputs(
    p: ModelPipeline,
    target_portion: float,
) -> None:
    # Target portion needs to be a float between zero and one
    if target_portion < 0:
        raise ValueError(
            "Target portion needs to be a value greater than zero, "
            f"given {target_portion}"
        )
    if target_portion > 1:
        raise ValueError(
            "Target portion needs to be a value less or equal to one, "
            f"given {target_portion}"
        )

    # Confirm the model pipeline is in the correct state
    Checker.columns_were_renamed(p)
    Checker.timestamp_required(p)
    Checker.unique_ids_were_created(p)


def build_tensors(p: ModelPipeline) -> tuple[Tensor, Tensor, Tensor]:
    """
    Step one: build edge index and label tensors
    """
    edge_index = torch.tensor(
        p.df[["from_account_idx", "to_account_idx"]].values.T,
        dtype=torch.long,
    )
    edge_label = torch.tensor(p.df["is_laundering"].values, dtype=torch.long)
    edge_to_row = torch.arange(len(p.df))  # maps edge_index column to DataFrame row

    return edge_index, edge_label, edge_to_row


def identify_illicit_edges_nodes(
    edge_index: Tensor,
    edge_label: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Step two: identify illicit edges and their nodes
    """
    illicit_edge_mask = edge_label == 1
    illicit_edge_indices = illicit_edge_mask.nonzero(as_tuple=False).view(-1)
    illicit_nodes = edge_index[:, illicit_edge_indices].flatten().unique()

    return illicit_edge_mask, illicit_nodes


def extract_illicit_neighborhood(
    p: ModelPipeline,
    target_portion: float,
    input_file: str,
    illicit_nodes: Tensor,
    edge_index: Tensor,
    edge_to_row: Tensor,
) -> tuple[pd.DataFrame, Tensor]:
    """
    Step three: extract k-hop neighborhood around illicit nodes. Uses
    a simple search routine to avoid the case where the neighborhood is
    larger than the target graph portion
    """

    # Subjectively target the illicit neighborhood to represent between
    # 40% and 70% of the resultant graph
    tolerance_low = 0.4 * target_portion * len(p.df)
    tolerance_high = 0.7 * target_portion * len(p.df)

    min_hops = 0
    max_hops = 20
    best_hops = None
    while min_hops <= max_hops:
        mid_hops = (min_hops + max_hops) // 2

        # For the current hops centerpoint, create a tentative subgraph
        _, _, _, edge_mask = k_hop_subgraph(
            node_idx=illicit_nodes,
            num_hops=mid_hops,
            edge_index=edge_index,
            relabel_nodes=False
        )
        illicit_neighbor_edge_indices = edge_mask.nonzero(as_tuple=False).view(-1)
        illicit_df_indices = edge_to_row[illicit_neighbor_edge_indices]
        illicit_subgraph_size = len(illicit_df_indices)
        illicit_subgraph_df = p.df.iloc[illicit_df_indices.tolist()].copy()

        logging.info(
            f"Tried num_hops={mid_hops}: "
            f"illicit subgraph size={illicit_subgraph_size}"
        )
        logging.info(
            f" - Percent of {input_file}: "
            f"{round(len(illicit_subgraph_df) / len(p.df) * 100, 3)} %"
        )

        if tolerance_low <= illicit_subgraph_size <= tolerance_high:
            # This 'num_hops' is good enough
            best_hops = mid_hops
            break
        elif mid_hops == 0:
            # This is the smallest graph possible, might just have to
            # use it
            best_hops = mid_hops
            break
        elif illicit_subgraph_size < tolerance_low:
            # Need a bigger neighborhood
            min_hops = mid_hops + 1
        else:
            # Need a smaller neighborhood
            max_hops = mid_hops - 1

    # Ensure we found a best hops value
    if best_hops is None:
        raise Exception(
            "Could not find an exact 'num_hops' fitting the defined "
            "tolerance. Need to be more tolerant, or choose a "
            "different target portion."
        )

    logging.info(f"Selected num_hops={best_hops} for illicit neighborhood")
    return illicit_subgraph_df, illicit_df_indices


def expand_graph_licitly(
    p: ModelPipeline,
    target_portion: float,
    illicit_subgraph_df: pd.DataFrame,
    illicit_df_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Step four: sample additional licit edges to reach target portion
    """
    target_total = int(target_portion * len(p.df))
    needed = max(0, target_total - len(illicit_subgraph_df))
    # Exclude already-included edges
    licit_df = p.df[(p.df["is_laundering"] == 0) & (~p.df.index.isin(illicit_df_indices.tolist()))].copy()
    logging.info(f"Total available licit edges: {len(licit_df)}")
    # Sample a portion of needed licit edges (oversample a bit for neighborhood coverage)
    sample_size = min(int(needed / 60), len(licit_df))  # adjustable oversampling factor
    sampled_licit_df = licit_df.sample(n=sample_size, random_state=42)
    sampled_licit_indices = sampled_licit_df.index.tolist()
    sampled_licit_edge_indices = torch.tensor(sampled_licit_indices, dtype=torch.long)
    # Get all involved nodes for the sampled licit edges
    sampled_node_ids = np.unique(np.concatenate([
        sampled_licit_df["from_account_idx"].values,
        sampled_licit_df["to_account_idx"].values
    ]))
    sampled_node_tensor = torch.tensor(sampled_node_ids, dtype=torch.long)
    logging.info(
        f"Sampled licit edges: {len(sampled_licit_indices)} | "
        f"Nodes involved: {len(sampled_node_ids)}"
    )
    return sampled_node_tensor, sampled_licit_edge_indices


def sample_licit_neighborhood(
    edge_index: Tensor,
    sampled_node_tensor: Tensor,
    sampled_licit_edge_indices: Tensor,
) -> Tensor:
    """
    Step five: extract k-hop neighborhood around sampled licit nodes
    """
    _, _, _, sampled_edge_mask = k_hop_subgraph(
        node_idx=sampled_node_tensor,
        num_hops=2,
        edge_index=edge_index,
        relabel_nodes=False
    )
    sampled_neighbor_edge_indices = sampled_edge_mask.nonzero(as_tuple=False).view(-1)
    # Combine with sampled licit edges explicitly (in case any are dropped in k-hop)
    sampled_combined_edge_indices = torch.unique(torch.cat([
        sampled_neighbor_edge_indices,
        sampled_licit_edge_indices
    ]))
    return sampled_combined_edge_indices


def combine_illicit_licit_subgraphs(
    p: ModelPipeline,
    illicit_df_indices: Tensor,
    sampled_combined_edge_indices: Tensor,
) -> pd.DataFrame:
    """
    Step six: combine illicit subgraph with sampled licit subgraph
    """
    final_df_row_indices = torch.unique(torch.cat([
        illicit_df_indices,
        sampled_combined_edge_indices
    ]))
    final_subset_df = p.df.iloc[final_df_row_indices.tolist()].copy()
    logging.info(f"Final subset size: {len(final_subset_df)}")
    logging.info(f"Percent of HI-Small: {round(len(final_subset_df) / len(p.df) * 100, 3)} %")
    logging.info(f"Total transactions: {len(final_subset_df)}")
    logging.info(
        "Percent of illicit transactions: "
        f"{round(len(final_subset_df[final_subset_df['is_laundering'] == 1]) / len(final_subset_df) * 100, 3)}"
        " %"
    )
    return final_subset_df


def reformat_final_df(p: ModelPipeline, final_subset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step seven: return data to the original IBM format for saving
    """
    inv_map = {v: k for k, v in p.column_mapping.items()}
    final_subset_df.rename(columns=inv_map, inplace=True)
    logging.info(final_subset_df.columns)
    final_subset_df["Account"] = final_subset_df["from_account_id"].str.split("_").str[1]
    final_subset_df["Account.1"] = final_subset_df["to_account_id"].str.split("_").str[1]
    final_subset_df.drop(columns=["from_account_id", "to_account_id","from_account_idx","to_account_idx"], inplace=True)
    final_subset_df = final_subset_df[p.ibm_column_header]
    logging.info(final_subset_df.columns)
    return final_subset_df


def create_subgraph(
    p: ModelPipeline,
    save_subgraph: bool=False,
    file_name: str="subset_transactions.csv",
    target_portion: float=0.25
) -> None:
    input_file = p.dataset_path.split('/')[-1]
    validate_subgraph_inputs(p, target_portion)

    # Create the subgraph step by step.
    # Refactored directly from `Code/notebooks/Sophie/create_subgraph.ipynb`
    edge_index, edge_label, edge_to_row = build_tensors(p)
    illicit_edge_mask, illicit_nodes = identify_illicit_edges_nodes(edge_index, edge_label)
    logging.info(
        f"Percent illicit in {input_file}: "
        f"{round(illicit_edge_mask.sum().item() / len(p.df) * 100, 3)}%"
    )
    illicit_subgraph_df, illicit_df_indices = extract_illicit_neighborhood(
        p=p,
        target_portion=target_portion,
        input_file=input_file,
        illicit_nodes=illicit_nodes,
        edge_index=edge_index,
        edge_to_row=edge_to_row,
    )
    logging.info(f"Illicit subgraph size: {len(illicit_subgraph_df)}")
    logging.info(
        f"Percent of {input_file}: "
        f"{round(len(illicit_subgraph_df) / len(p.df) * 100, 3)} %"
    )
    sampled_node_tensor, sampled_licit_edge_indices = expand_graph_licitly(
        p=p,
        target_portion=target_portion,
        illicit_subgraph_df=illicit_subgraph_df,
        illicit_df_indices=illicit_df_indices,
    )
    sampled_combined_edge_indices = sample_licit_neighborhood(
        edge_index=edge_index,
        sampled_node_tensor=sampled_node_tensor,
        sampled_licit_edge_indices=sampled_licit_edge_indices,
    )
    final_subset_df = combine_illicit_licit_subgraphs(
        p=p,
        illicit_df_indices=illicit_df_indices,
        sampled_combined_edge_indices=sampled_combined_edge_indices,
    )
    final_subset_df = reformat_final_df(p, final_subset_df)

    # Provides an estimate of file size. Useful when creating a graph
    # that will be saved in a location where size matters, e.g. when
    # creating a test graph to store in GitHub
    estimated_size_mb = estimate_df_to_csv_size(final_subset_df) / (1024 * 1024)
    logging.info(f"Estimated output file size: {estimated_size_mb:.2f} MB")

    if save_subgraph:
        final_subset_df.to_csv((os.path.join(p.data_dir, file_name)), index=False)
