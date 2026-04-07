import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run single-GPU BFS with UVM")
    parser.add_argument("--dataset", type=str, required=True, help="path to graph dataset")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import rmm
    import cudf
    import cugraph

    pool_size_bytes = 10 * (2**30)
    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=False,
        initial_pool_size=pool_size_bytes
    )
    print(f"GPU {args.gpu} allocation completed. UVM ON (pool={pool_size_bytes / (2**30):.0f}GB)")

    
    print(f"Loading dataset... ({args.dataset})")
    e_list = cudf.read_csv(
        args.dataset,
        delimiter=' ',
        names=["src", "dst"],
        dtype=["int32", "int32"],
        header=None
    )

    if len(e_list) == 0:
        raise ValueError("Parsed edge list is empty. Check the dataset file.")

    print(f"Edges loaded: {len(e_list)}")

    G = cugraph.Graph() 
    G.from_cudf_edgelist(
        e_list,
        source="src",
        destination="dst",
        # renumber=True
    )

    if args.loop:
        while True:
            t_start = time.time()
            df = cugraph.bfs(G, 1)
            print("Out:", time.time() - t_start)
    else:
        t_start = time.time()
        df = cugraph.bfs(G, 1)
        print("Out:", time.time() - t_start)
        print(df.head())


if __name__ == "__main__":
    main()