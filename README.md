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

### Conclusion

The key-value cache in Transformer models is a powerful technique to enhance the efficiency of the attention mechanism during inference. By reusing previously computed keys and values, the model can significantly reduce the computational overhead and speed up the generation process. This technique is particularly useful in applications requiring real-time or low-latency processing, such as machine translation, text generation, and other NLP tasks.

## 2. Multi-Query Attention and Grouped-query Attention 

[(Video)](https://www.youtube.com/watch?v=pVP0bu8QA2w) 

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

### Conclusion

Multi-query attention is a variant of the attention mechanism designed to improve efficiency in Transformer models. By sharing the Key and Value matrices across all attention heads while keeping separate Query matrices, it reduces memory usage and computational overhead. This makes it especially useful in scenarios where model efficiency is critical, such as during inference in large-scale models. Despite some tradeoffs in expressive power, multi-query attention offers a scalable and efficient alternative to standard multi-head attention.


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

### Conclusion

Grouped Query Attention in Transformer models is an optimization technique that balances the flexibility of multi-head attention with the efficiency of multi-query attention. By grouping attention heads and sharing key and value vectors within each group, this approach reduces memory usage and computational overhead while still allowing for diverse attention patterns. This makes it a valuable technique for scaling Transformer models, especially in resource-constrained environments.

## 3. Fast LLM Serving with vLLM and PagedAttention
[(Video)](https://www.youtube.com/watch?v=5ZlavKF_98U)
