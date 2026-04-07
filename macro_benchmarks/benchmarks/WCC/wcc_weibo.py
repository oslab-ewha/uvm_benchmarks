import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run single-GPU BFS with UVM")
    parser.add_argument("--dataset", type=str, required=True, help="path to graph dataset")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import rmm
    import cudf
    import cugraph

    pool_size_bytes = 10 * (2**30)
    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=True,
        initial_pool_size=pool_size_bytes
    )
    print(f"GPU 0 allocation completed. UVM ON (pool={pool_size_bytes / (2**30):.0f}GB)")

    print(f"Loading dataset... ({args.dataset})")
    e_list = cudf.read_csv(
        args.dataset,
        # delim_whitespace=True,
        delimiter=' ',
        names=["src", "dst"],
        dtype=["int32", "int32"],
        header=None
    )

    if len(e_list) == 0:
        raise ValueError("Parsed edge list is empty. Check the dataset file.")

    print(f"Edges loaded: {len(e_list)}")

    G = cugraph.Graph() #directed=True)
    G.from_cudf_edgelist(
        e_list,
        source="src",
        destination="dst",
        # renumber=True
    )

    # start_external = int(e_list["src"].iloc[0])
    # start_df = cudf.DataFrame({"src": [start_external]})
    # start_internal = int(G.lookup_internal_vertex_id(start_df, "src").iloc[0])

    # print(f"Start vertex (external): {start_external}")
    # print(f"Start vertex (internal): {start_internal}")

    if args.loop:
        while True:
            t_start = time.time()
            df = cugraph.weakly_connected_components(G)
            print("Out:", time.time() - t_start)
    else:
        t_start = time.time()
        df = cugraph.weakly_connected_components(G)
        print("Out:", time.time() - t_start)
        print(df.head())


if __name__ == "__main__":
    main()