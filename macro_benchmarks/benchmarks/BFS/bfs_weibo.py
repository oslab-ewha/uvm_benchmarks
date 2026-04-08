import os
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run single-GPU BFS with UVM")
    parser.add_argument("--dataset", type=str, required=True, help="path to graph dataset")
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument("--reserve-gb", type=float, default=0.0,
                        help="GPU device memory to reserve in GB before graph construction")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import rmm
    import cudf
    import cugraph
    import cupy as cp

    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=False
    )

    print(f"GPU {args.gpu} ready. UVM ON, pool_allocator=False")

    # Allocate extra gpu memory (for memory oversubscription test)
    reserved_mem = None
    if args.reserve_gb > 0:
        reserve_bytes = int(args.reserve_gb * (2**30))
        reserved_mem = cp.cuda.Memory(reserve_bytes)
        print(f"Reserved GPU device memory: {args.reserve_gb:.2f} GB")

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

    # reserved_mem is freed automatically when program exits / reference is dropped


if __name__ == "__main__":
    main()