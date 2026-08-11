# Run one deterministic MUCC v1 subsample with the isolated FlashWeave runtime.
# Arguments: expression TSV, conditioning metadata TSV, output edgelist.
using FlashWeave

if length(ARGS) != 3
    error("expected: expression_tsv metadata_tsv output_edgelist")
end

results = learn_network(
    ARGS[1],
    ARGS[2],
    sensitive=true,
    heterogeneous=false,
    FDR=true,
    max_k=3,
    normalize=true,
    track_rejections=false,
    verbose=false,
)
save_network(ARGS[3], results, detailed=true)
