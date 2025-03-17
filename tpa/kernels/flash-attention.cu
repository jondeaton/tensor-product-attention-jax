// WORK IN PROGRESS: Flash attention 2 CUDA.
// Plans to support non-materializing Tensor Product Attention kernel soon.

#include <cuda.h>
#include <float.h>
#include <mma.h>
#include <stdio.h>

#define WARP_SIZE 32
#define CDIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

__device__ __forceinline__ float _lse_accum(float la, float lb) {
  float max = la > lb ? la : lb;
  float min = la > lb ? lb : la;
  return max + logf(1 + expf(min - max));
}

// Accumulates log-sum-exp across threads in a warp.
__device__ __forceinline__ float _warp_accum_lse(float lse, int width) {
  for (int offset = 1; offset <= width / 2; offset *= 2) {
    lse = _lse_accum(lse, __shfl_down_sync(0xffffffff, lse, offset));
  }
  return lse;
}

// WMMA tile sizes. WMMA only supports fixed sizes; for half precision these are
// typically 16×16×16.
#define WARP_MMA_BLOCK 16

// Kernel that multiplies a Q tile (size: [WMMA_M, d]) with a K tile (size:
// [WMMA_N, d]) to produce a logits tile (size: [WMMA_M, WMMA_N]) as:
// logits_tile = Q_tile * K_tile^T. In this example we assume that Q and K (in
// half precision) are stored in "global" memory. In your flash attention kernel
// you would likely first load these into shared memory.
__global__ void flash_attention_wmma_kernel(
    const __half *__restrict__ q, // query tile: (WARP_MMA_BLOCK, d)
    const __half *__restrict__ k, // keys tile: (WARP_MMA_BLOCK, d)
    float *__restrict__ logits,   // (WARP_MMA_BLOCK, WARP_MMA_BLOCK)
    int d, // contraction dimension (must be a multiple of WMMA_K)
) {
  // Declare the accumulator fragment and initialize to zero.
  constexpr int WMMA_M, WMMA_N, WMMA_K = WARP_MMA_BLOCK;

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  // Loop over the shared dimension in steps of WMMA_K.
  // Each iteration multiplies a [WMMA_M x WMMA_K] fragment from Q with a
  // [WMMA_N x WMMA_K] fragment from K.
  for (int i = 0; i < d; i += WMMA_K) {
    // Declare fragments for Q and K.
    // For Q we use row_major, for K we use col_major (so that loading from K
    // yields the transposed behavior).
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                   wmma::col_major>
        b_frag;

    // Compute the pointer for this sub-tile.
    // Q is assumed to be stored in row-major order.
    // Each Q tile starts at row = tile_q_idx * WMMA_M and column = i.
    const __half *a_tile_ptr = q + tile_q_idx * WMMA_M * ld_q + i;
    // Similarly, K is stored row-major but we want K^T.
    // We load K using col_major order. Each K tile starts at row = tile_k_idx *
    // WMMA_N and column = i.
    const __half *b_tile_ptr = k + tile_k_idx * WMMA_N * ld_k + i;

    // Load the fragments from memory.
    // The second argument is the leading dimension (pitch) of the matrix.
    wmma::load_matrix_sync(a_frag, a_tile_ptr, ld_q);
    wmma::load_matrix_sync(b_frag, b_tile_ptr, ld_k);

    // Perform the matrix multiply-accumulate.
    // This computes: c_frag = a_frag * b_frag + c_frag.
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // Compute the output pointer.
  // We assume that the full logits matrix is stored in row-major order with
  // dimensions:
  //   [num_tile_rows * WMMA_M, num_tile_cols * WMMA_N].
  // The stride (pitch) of a full row is num_tile_cols * WMMA_N.
  int num_tile_rows = gridDim.x; // equals the number of Q tiles
  float *c_tile_ptr = logits + tile_q_idx * WMMA_M * (num_tile_cols * WMMA_N) +
                      tile_k_idx * WMMA_N;

  // Store the result from the accumulator fragment to global memory.
  // The second argument to store_matrix_sync is the leading dimension of the
  // destination matrix.
  wmma::store_matrix_sync(c_tile_ptr, c_frag, num_tile_cols * WMMA_N,
                          wmma::mem_row_major);
}

void _load_block(          //
    __half *u_smem,        // SMEM pointer shaped (block_size, dim)
    __half *u,             // Global pointer to block start.
    int block_size,        // size of block
    int dim,               // dimension of block
    int threads_per_block, // number of threads per block.
) {
  // TODO: load with dim chunks in parallel across  threads.
  int threads_per_line = threads_per_block / block_size;
  int i = threadIdx.x / threads_per_line;
  if (threadIdx.x % threads_per_line == 0) {
    for (int j = 0; j < dk; j++) {
      u_smem[i * dk + j] = u[dk * i + j];
    }
  }
}

template <                       //
    const int q_block_size,      // queries processed by each thread block
    const int kv_block_size,     // keys processed by each thread block
    const int dk,                // key dimension
    const int dv,                // value dimension
    const int threads_per_block, // number of threads per thread block
    const int warps_per_block    // number of warps per thread block
    >                            //
__global__ void kernel(          //
    __half *q,                   // queries,  (batch_size, q_len, dk)
    __half *k,                   // keys,  (batch_size, kv_len, dk)
    __half *v,                   // values,  (batch_size, kv_len, dv)
    __half *o,                   // output, (batch_size, q_len, dv)
    __half *l,                   // log sum exp (batch_size, q_len)
    int batch_size,              // size of batch (heads flattened)
    int q_len,                   // number of queries
    int kv_len,                  // number of keys and values
    __half sm_scale,             // softmax scale
) {

  // Thread block: (batch, q_block)
  int q_block_offset =
      blockIdx.x * (q_len * q_block_size * dk) + blockIdx.y * q_block_size * dk;

  int warp_id = threadIdx.x / WARP_SIZE;
  int warp_thread_id = threadIdx.x % WARP_SIZE;

  int q_warp_block_size = q_block_size / warps_per_block;
  int q_warp_offset = q_block_offset + warp_id * q_warp_block_size;

  extern __shared__ __half q_smem[q_block_size][dk];
  extern __shared__ __half o_smem[q_block_size][dv];
  extern __shared__ __half l_smem[q_block_size];

  int dk_block = CDIV(q_warp_block_size * dk, WARP_SIZE);

  _load_block(q_smem, q + q_block_offset, block_size, dim, threads_per_block);

  int threads_per_line = threads_per_block / q_block_size;
  int i = threadIdx.x / threads_per_line;
  if (threadIdx.x % threads_per_line == 0) {
    l_smem[i] = -INFINITY;
    // Initialize output block accumulation.
    for (int j = 0; j < dv; j++) {
      o_smem[i][j] = 0;
    }
  }

  extern __shared__ __half k_smem[kv_block_size][dk];
  extern __shared__ __half v_smem[kv_block_size][dv];
  extern __shared__ __half x_smem[q_block_size][kv_block_size];

  int num_kv_blocks = CDIV(kv_len, kv_block_size);
  for (                               //
      int kv_block_index = 0;         //
      kv_block_index < num_kv_blocks; //
      kv_block_index++                //
  ) {
    int k_block_offset = kv_block_index * kv_block_size * dk;
    int v_block_offset = kv_block_index * kv_block_size * dv;
    _load_block(k_smem, k + k_block_offset, kv_block_size, dk,
                threads_per_block);
    _load_block(v_smem, v + v_block_offset, kv_block_size, dv,
                threads_per_block);

    // Perform query/key inner products matmul.
    int warp_block_size = q_block_size / warps_per_block;
    assert warp_block_size == WARP_MMA_BLOCK;
    for (int tile_index = 0; tile_index < warp_per_block, tile_index++) {
      // NOTE: this shit is why we fucking invented compilers
      flash_attention_wmma_kernel(                                    //
          q_smem + warp_id * warp_block_size * dk,                    //
          k_smem + tile_index * warp_block_size * dk,                 //
          x_smem +                                                    //
              kv_block_size * warp_id + warp_block_size * tile_index, // logits
          dk                                                          //
      );

      // TODO: handle lse and reduction for the warp
      // l_smem[];
      // compute log-sum-exp across block of keys
      // __half l = _warp_accum_lse(x, WARP_SIZE);
      // extern __shared__ __half l_warp[WARP_SIZE]; // for warp reduction
    }

    // TODO: warps are grouped on the innermost thread dimension??
    if (threadIdx.y % WARP_SIZE == 0) {
      l_warp[threadIdx.y / WARP_SIZE] = l;
    }
    __syncthreads();

    // accumulate lses from each warp.
    int num_warps = CDIV(kv_block_size, WARP_SIZE);
    if (threadIdx.y == 0) {
      float lse = -INFINITY;
      for (int w = 0; w < num_warps; w++) {
        lse = _lse_accum(lse, lse_shared[w]);
      }
      l_next_smem[threadIdx.x] = lse;
    }
    __syncthreads();

    __half lse = _lse_accum(lse, lse_next);
    __half p = expf(x - lse);

    // previous output correction.
    __half c = expf(lse - lse_next);

    // compute output for single block.
    __half o[dv];
    __shared__ __half o_warp[num_warps][dv];
    for (int i = 0; i < dv; i++) {
      __half oi = _warp_accum_add(p * v[i], WARP_SIZE);
      if (threadIdx.y % WARP_SIZE == 0) {
        o_warp[threadIdx.y / WARP_SIZE] = oi;
      }
      __syncthreads();
      __half o = 0;
      for (int w = 0; w < num_warps) {
        o += o_warp[w][i];
      }
      __syncthreads();
    }

    for (int i = 0; i < dv; i++) {
      o_smem[i] = o_smem[i] * c + o
    }
    //????

    // so far in pallas:
    // qk = q @ k.t
    // logsumexp(qk, axis=-1)

    // ugh... dude fuck

    // things...
  }
}

int main() {
  int batch_size = 16;
  int q_seq_len = 512;
  int kv_seq_len = 512;
  int d_head = 64;

  int q_size = batch_size * q_seq_len * d_head;
  int kv_size = batch_size * kv_seq_len * d_head;
  int l_size = batch_size * q_seq_len * d_head;

  __half *q, *k, *v, *o, *l;
  q = (__half *)malloc(q_size * sizeof(__half));
  k = (__half *)malloc(kv_size * sizeof(__half));
  v = (__half *)malloc(kv_size * sizeof(__half));
  o = (__half *)malloc(q_size * sizeof(__half));
  l = (__half *)malloc(lm_size * sizeof(__half));

  // fill with some random data.
  for (int i = 0; i < q_size; i++) {
    q[i] = __float2half(i % 5 - 10);
  }
  for (int i = 0; i < kv_size; i++) {
    k[i] = __float2half(i % 4 - 8);
    v[i] = __float2half(i % 3 - 6);
  }

  int q_block_size = 64;
  int kv_block_size = 64;

  int num_q_blocks = cdiv(q_seq_len, q_block_size);
  int num_kv_blocks = cdiv(kv_seq_len, kv_block_size);

  // (lq, lk) + q + o + k + v + lse
  const int smem_size = sizeof(__half) *                   //
                        ((q_block_size * kv_block_size)    // logits
                         + (2 * q_block_size * head_embd)  // q/o
                         + (2 * kv_block_size * head_embd) // kv
                         + q_block_size                    // lse
                        );

  int max_sram_size;
  cudadevicegetattribute(&max_sram_size, cudadevattrmaxsharedmemoryperblock, 0);
  printf("max shared memory: %d, requested shared memory: %d \n", max_sram_size,
         smem_size);

  // todo: copy data to/from device

  // In Flash Attention 2 he says that they use 4 or 8 warps pre thread block.
  // But at 29:11 of the ThunderKittens talk, they say that you generally want
  // two (at least?) warps per SM quadrant. An SM quadrant appeasrs to be a
  // physical piece of hardware in which all the threads of a warp are
  // co-scheduled onto and has an associated TensorCore. There appear to be 4 SM
  // quadrants which all share the same L1 / Shared Memory lane.
  // Hence we want 4 SM quadrants x 2 warps per quadrant = 8 warps per threads
  // block.
  int warps_per_block = 8;
  int threads_per_block = WARP_SIZE * warps_per_block;
  int q_block_size = WARP_MMA_BLOCK * warps_per_block;

  dim3 grid_dim(batch_size, cdiv(q_seq_len, q_block_size));
  dim3 block_dim(threads_per_block);

  kernel<                                                      //
      q_block_size,                                            //
      kv_block_size,                                           //
      dk,                                                      //
      dv,                                                      //
      threads_per_block,                                       //
      warps_per_block                                          //
      ><<<grid_dim, block_dim, smem_size>>>(                   //
      q, k, v, seq_len, head_embd, tr, tc, softmax_scale, l, o //
  );
}
