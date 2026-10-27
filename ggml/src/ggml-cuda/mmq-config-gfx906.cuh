// gfx906 (Vega20 / MI50, wave64) MMQ config.
//
// The generic GCN table predates the TQ4_1S weight type and has no entries for
// it, so a gfx906 host would get GGML_TYPE_COUNT for the native TQ4 MMQ path.
// Keep the GCN table as the fallback for every existing type and add a TQ4
// table with the same tile geometry: the dp4a tile loader expands
// each 4-bit TQ4 block to int8 centroids, and stock MMQ then runs
// v_dot4_i32_i8 through ggml_cuda_dp4a().
//
// I=128 matches the TQ4 loader's two 64-element halves.  J is capped at 64:
// gfx906 has no MFMA/WMMA, and the per-wave dp4a tile is already register
// heavy; larger J only helps full-row prefill, not the MoE / row-sharded
// shapes where this path matters most.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gfx906(ggml_type type, int J, bool fallback) {
    if (type == GGML_TYPE_TQ4_1S) {
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true);

        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128,  8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
        CASE(GGML_TYPE_TQ4_1S, 512, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false);
    }
    return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
}
