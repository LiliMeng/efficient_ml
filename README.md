# Efficient Transformer Inference

## 1. What's the problem? The KV Cache: Memory Usage in Transformers 
[(Video)](https://www.youtube.com/watch?v=80bIUggRJf4&t=324s)

In Transformer models, the key-value cache is a technique used to optimize the efficiency of the attention mechanism, particularly during inference. Here’s a detailed explanation of what it is and how it works:

### Key-Value Cache in Transformer Models

#### Background: Transformer Architecture

Transformers use a mechanism called self-attention to weigh the importance of different tokens in a sequence relative to each other. In a typical Transformer layer, the self-attention mechanism involves:

1. **Query (Q)**: Represents the current token we are focusing on.
2. **Key (K)**: Represents all tokens in the sequence and is used to calculate attention scores.
3. **Value (V)**: Represents all tokens in the sequence and is used to aggregate the final representation based on attention scores.

The attention mechanism calculates the relevance of each token (key) to the current token (query) and uses this relevance to weigh the values. This is mathematically represented as:

<img width="344" alt="Screenshot 2024-07-15 at 11 26 54 AM" src="https://github.com/user-attachments/assets/195faf04-c9bb-47d5-b574-492adc898f86">

#### Key-Value Cache in Decoding

During the training phase, Transformers process the entire sequence at once. However, during inference (e.g., in tasks like machine translation or text generation), the model generates tokens one by one. To improve efficiency and reduce redundancy, key-value caching is used.

1. **Caching Mechanism**:
   - **Keys and Values Storage**: Instead of recalculating keys (K) and values (V) for the entire input sequence at each decoding step, the model stores these values in a cache after the first pass.
   - **Reusing Cached Values**: For subsequent decoding steps, the model only needs to compute the query (Q) for the new token and can reuse the cached keys and values from previous steps.

2. **Efficiency**:
   - **Reduced Computation**: By reusing the cached keys and values, the model avoids redundant calculations, significantly speeding up the inference process.
   - **Memory Management**: Efficient management of the cache ensures that the model doesn't need to store the entire sequence repeatedly, optimizing memory usage.

#### Example Workflow

Here’s a step-by-step workflow illustrating the key-value caching mechanism during inference:

1. **Initial Decoding Step**:
   - Input: "The cat"
   - Compute Q, K, V for each token in "The cat".
   - Store the computed K and V in the cache.

2. **Subsequent Decoding Step**:
   - Input: "The cat sat"
   - Compute Q for the new token "sat".
   - Retrieve K and V from the cache for "The cat".
   - Compute attention using the new Q and the cached K, V.

3. **Continue Decoding**:
   - Each new token is processed by computing its query (Q), while reusing the cached keys and values from all previous tokens.

### Practical Considerations

1. **Cache Initialization**: At the start of decoding, the cache is initialized and populated with the keys and values for the initial sequence.
2. **Cache Update**: As new tokens are generated, the cache is updated with the keys and values for these tokens.
3. **Memory Management**: Efficient management and updating of the cache are crucial to ensure that the model remains efficient and scalable.

### Tradeoffs

1. **Memory Usage**:
   - **Pros**: Significantly reduces the computational load during inference.
   - **Cons**: Requires additional memory to store the cached keys and values.

2. **Complexity**:
   - **Pros**: Simplifies the attention computation in each step by avoiding redundant calculations.
   - **Cons**: Adds complexity to the model implementation due to the need to manage the cache.

## 2. Multi-Query Attention and Grouped-query Attention 

[(Video)](https://www.youtube.com/watch?v=pVP0bu8QA2w) 

High level overview:
<img width="1408" alt="Screenshot 2024-07-15 at 3 39 25 PM" src="https://github.com/user-attachments/assets/15abd92b-578b-4af0-bdc5-db2d20a8c75f">


   ### 2.1 Multi-Query Attention in Transformers

Original Paper: Fast Transformer Decoding: One Write-Head is All You Need [(link)](https://arxiv.org/pdf/1911.02150)

Multi-query attention is a variant of the attention mechanism used in Transformer models to improve computational efficiency during the decoding phase. The primary goal of multi-query attention is to reduce the memory and computation overhead associated with the self-attention mechanism, especially in large models.

#### Background: Standard Multi-Head Attention

In the standard Transformer architecture, multi-head attention is used to allow the model to focus on different parts of the input sequence simultaneously. This involves:

1. **Multiple Heads**: Each head performs its own self-attention operation using separate sets of Query (Q), Key (K), and Value (V) matrices.
2. **Parallel Attention**: The results from each head are concatenated and linearly transformed to produce the final output.

<img width="506" alt="Screenshot 2024-07-15 at 11 14 56 AM" src="https://github.com/user-attachments/assets/375c7b45-7b0b-4c04-a077-7f252590b16a">

This process requires computing and storing separate K and V matrices for each attention head, which can be resource-intensive.

#### Multi-Query Attention

Multi-query attention aims to reduce this resource usage by sharing the Key (K) and Value (V) matrices across all attention heads, while still allowing each head to have its own Query (Q) matrix.

1. **Single Key and Value**: Instead of having separate K and V matrices for each head, multi-query attention uses a single shared K and V matrix for all heads.
2. **Separate Queries**: Each head retains its own Q matrix, allowing for different attention patterns across heads.

#### Formula for Multi-Query Attention
<img width="551" alt="Screenshot 2024-07-15 at 11 26 38 AM" src="https://github.com/user-attachments/assets/bcf50b4e-5d86-4f16-9b8a-1dfa9f4224a1">



#### Benefits of Multi-Query Attention

1. **Reduced Memory Usage**:
   - Sharing K and V matrices across all heads reduces the memory required to store these matrices.
   - This is particularly beneficial in the decoder, where multiple layers of self-attention can lead to significant memory consumption.

2. **Improved Computation Efficiency**:
   - By reducing the number of K and V matrices, the computational load during the attention computation is decreased.
   - This can lead to faster inference times, which is crucial for real-time applications.

3. **Scalability**:
   - Multi-query attention allows for scaling up the number of attention heads without a corresponding linear increase in memory and computation costs.
   - This enables the use of more attention heads, potentially improving the model's ability to capture complex dependencies in the data.

#### Tradeoffs

1. **Expressive Power**:
   - Sharing K and V matrices may limit the diversity of attention patterns that different heads can learn compared to having separate K and V matrices.
   - However, in practice, the impact on performance can be mitigated by the increased number of heads and the shared Q matrices' flexibility.

2. **Implementation Complexity**:
   - While conceptually simpler in some respects, multi-query attention requires careful implementation to ensure that the shared K and V matrices are effectively utilized across all heads.

### 2.2 Grouped Query Attention in Transformer Models
Paper: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints [(link)](https://arxiv.org/pdf/2305.13245)

Grouped Query Attention is an advanced technique used in Transformer models to strike a balance between the flexibility of multi-head attention and the computational efficiency of multi-query attention. The idea is to partition the query heads into groups, where each group shares a common set of key and value pairs.

#### Background: Standard Multi-Head Attention

In a standard multi-head attention mechanism, each head independently computes its own set of query (Q), key (K), and value (V) vectors. This allows the model to focus on different parts of the input sequence simultaneously, providing a richer representation of the data. However, it also means that each head requires separate computation and storage of Q, K, and V, which can be resource-intensive.

#### Concept of Grouped Query Attention

Grouped Query Attention aims to reduce the computational and memory overhead by grouping the query heads and sharing the key and value vectors within each group. Here's how it works:

1. **Grouping Heads**: Instead of having independent heads, the attention heads are divided into groups. Each group shares the same key (K) and value (V) vectors.
2. **Separate Queries**: Within each group, the heads have separate query (Q) vectors, allowing for some diversity in attention patterns, but less so than in completely independent heads.

#### How Grouped Query Attention Works

1. **Input Representation**:
   - Assume we have `h` attention heads and we divide them into `g` groups, with each group containing `h/g` heads.
   - Let `Q`, `K`, and `V` be the sets of query, key, and value vectors respectively.

2. **Grouped Attention Calculation**:
   - For each group, compute a single set of key and value vectors.
   - Within each group, compute the attention scores using the separate query vectors for each head.

3. **Attention Formula**:
   - Compute attention for each group `i` as follows:
     \[
     \text{head}_{i,j} = \text{Attention}(Q_{i,j}, K_i, V_i)
     \]
     where \( Q_{i,j} \) is the query vector for the `j-th` head in the `i-th` group, and \( K_i \) and \( V_i \) are the shared key and value vectors for the `i-th` group.

4. **Concatenation and Output**:
   - Concatenate the outputs of all heads across all groups.
   - Apply a linear transformation to produce the final output.

#### Example Workflow

1. **Grouping and Initialization**:
   - Divide `h` heads into `g` groups.
   - Initialize separate Q vectors for each head, and shared K and V vectors for each group.

2. **Attention Computation**:
   - For each group:
     - Compute K and V vectors.
     - For each head in the group, compute the attention scores and weighted sum using the group's K and V.

3. **Combine Results**:
   - Concatenate the results from all heads.
   - Apply a final linear layer to integrate the information.

### Benefits of Grouped Query Attention

1. **Reduced Memory Usage**:
   - By sharing K and V vectors within groups, the memory required for storing these vectors is significantly reduced compared to fully independent heads.

2. **Computational Efficiency**:
   - The number of computations required for the attention mechanism is decreased since fewer unique K and V computations are needed.

3. **Flexible Representation**:
   - While reducing memory and computation, grouped query attention maintains some diversity in attention patterns by allowing separate Q vectors within each group.

### Tradeoffs

1. **Expressive Power**:
   - The sharing of K and V vectors within groups may reduce the model's ability to capture highly diverse attention patterns compared to completely independent heads.

2. **Implementation Complexity**:
   - Implementing grouped query attention requires careful management of the grouping logic and ensuring efficient computation across groups.

## 3. Fast LLM Serving with vLLM and PagedAttention
[(Video)](https://www.youtube.com/watch?v=5ZlavKF_98U)
Efficient Memory Management for Large Language Model Serving with PagedAttention [Paper](https://arxiv.org/pdf/2309.06180)

PagedAttention is a technique aimed at improving the efficiency and scalability of attention mechanisms, particularly in large language models like transformers. The primary goal of PagedAttention is to handle large sequences of data efficiently by leveraging paged memory management principles, similar to those used in computer systems for handling large data sets.

### Key Concepts and Principles

1. **Memory Paging**:
   - **Paging in Computing**: In computer systems, paging is a memory management scheme that eliminates the need for contiguous allocation of physical memory. It breaks memory into fixed-size blocks called pages, which can be loaded into any physical memory location.
   - **Paging in Attention Mechanisms**: PagedAttention applies a similar concept to the attention mechanism by breaking down the attention computation into smaller, more manageable chunks (pages), which can be processed independently and efficiently.

2. **Attention Mechanism**:
   - **Standard Attention**: In standard attention mechanisms, particularly in transformers, the computational complexity is \(O(n^2)\) where \(n\) is the sequence length. This is due to the need to compute attention scores for all pairs of input tokens.
   - **PagedAttention**: PagedAttention reduces this complexity by processing attention in smaller chunks, reducing the memory footprint and computational load at any given time.

### How PagedAttention Works

1. **Chunking the Input**:
   - The input sequence is divided into smaller chunks or pages. Each chunk contains a subset of the total input tokens, making the attention computation more manageable.
   
2. **Local Attention within Chunks**:
   - Attention is computed locally within each chunk. This means that tokens within a chunk attend to each other but not to tokens in other chunks initially. This reduces the computational complexity significantly.

3. **Cross-Chunk Attention**:
   - To maintain the global context, cross-chunk attention is performed. This can be done by allowing tokens at the boundaries of chunks to attend to tokens in adjacent chunks or by using a hierarchical attention mechanism where summaries of each chunk are used to perform global attention.

4. **Hierarchical Processing**:
   - PagedAttention can use hierarchical processing where each level of the hierarchy performs attention over progressively larger contexts, starting from local contexts (within chunks) to more global contexts (across chunks).

### Benefits of PagedAttention

1. **Scalability**:
   - By breaking down the attention computation into smaller chunks, PagedAttention scales better with longer input sequences, addressing the quadratic complexity issue of standard attention mechanisms.

2. **Efficiency**:
   - Memory and computational resources are used more efficiently, as only a subset of the data is processed at any given time. This can lead to faster training and inference times.

3. **Flexibility**:
   - PagedAttention can be adapted to different sizes of chunks and different strategies for cross-chunk attention, providing flexibility in balancing local and global context.



Here is a conceptual example of how PagedAttention might be implemented:

```python
import torch
import torch.nn as nn

class PagedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size):
        super(PagedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Divide the input sequence into chunks
        chunks = x.split(self.chunk_size, dim=0)
        
        # Compute local attention within each chunk
        local_attn_outputs = []
        for chunk in chunks:
            attn_output, _ = self.attention(chunk, chunk, chunk)
            local_attn_outputs.append(attn_output)
        
        # Concatenate the local attention outputs
        local_attn_outputs = torch.cat(local_attn_outputs, dim=0)
        
        # Compute cross-chunk attention (this is a simple form, more sophisticated methods can be used)
        global_context = torch.mean(local_attn_outputs, dim=0, keepdim=True)
        global_attn_output, _ = self.attention(local_attn_outputs, global_context, global_context)
        
        return global_attn_output

# Example usage:
embed_dim = 64
num_heads = 8
chunk_size = 16
paged_attention = PagedAttention(embed_dim, num_heads, chunk_size)

# Input sequence (batch_size, seq_length, embed_dim)
x = torch.rand(32, 128, embed_dim)
output = paged_attention(x)
print(output.shape)
```

In this example:
- The input sequence is divided into chunks of size `chunk_size`.
- Local attention is computed within each chunk.
- A simple form of cross-chunk attention is performed using the mean of local attention outputs as the global context.

PagedAttention provides a scalable and efficient approach to handle long sequences, making it a valuable technique in modern deep learning applications.

## 4. Longformer: An Efficient Transformer Variant
Longformer: The Long-Document Transformer [Paper](https://arxiv.org/pdf/2004.05150v2)
<img width="1318" alt="Screenshot 2024-07-15 at 4 57 21 PM" src="https://github.com/user-attachments/assets/5dfdb9ee-23f9-4f3b-a03b-97c113f1c27c">


Longformer is a Transformer model variant designed to handle long documents efficiently. Traditional Transformers have quadratic complexity with respect to the input sequence length, making them impractical for long sequences. Longformer addresses this limitation by introducing a sparse attention mechanism that scales linearly with sequence length.

### How Longformer Works

1. **Attention Mechanism**:
   - **Local Sliding Window Attention**: Each token attends to its neighbors within a fixed-size window. This is efficient and captures local context.
   - **Global Attention**: Important tokens, specified beforehand or learned, attend to all other tokens. This allows the model to capture long-range dependencies without incurring a quadratic cost.

2. **Combining Local and Global Attention**:
   - The model combines local and global attention by computing both and then integrating them. This ensures that the model benefits from both fine-grained local context and broader, global context.

3. **Implementation**:
   - The implementation involves masking out the irrelevant positions in the attention matrix, ensuring that each token only attends to its designated tokens, either locally or globally.

## 5.  Cross Layer KV-sharing

Reducing Transformer Key-Value Cache Size with Cross-Layer Attention [Paper](https://arxiv.org/pdf/2405.12981)
<img width="1153" alt="Screenshot 2024-07-16 at 10 28 10 AM" src="https://github.com/user-attachments/assets/0baedcd0-21ee-4ff3-80ab-68e822d1f827">
<img width="1153" alt="Screenshot 2024-07-16 at 10 31 12 AM" src="https://github.com/user-attachments/assets/76955436-316c-4a12-b98f-bf10b3a6beef">

## 6. Efficient KV Cache Reuse with RadixAttention

SGLang: Efficient Execution of Structured Language Model Programs [Paper](https://arxiv.org/pdf/2312.07104)

