# Optimizing LLM Training: An Overview of Techniques 👐 📚

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/m9v01CkHNjLKvt1eUHTWz.png)

Training large language models (LLMs) is both computationally intensive and time-consuming. However, optimizing the training process can significantly reduce costs, accelerate development, and enhance the final model's performance. This guide provides a comprehensive overview of optimization techniques, from model selection to fine-tuning the learning process.

## Optimization Methods

### 1. Memory and Computation Optimization
#### 1.1. Mixed Precision
Before diving into Mixed Precision and related topics, it’s crucial to understand what contributes to memory consumption during model training. A model consists of parameters, each represented as a real number stored in the computer's memory. Typically, these real numbers are stored in the float32 format, which requires 32 bits per number.

To put this in perspective, let's calculate the memory needed to load a model like Llama 70B. This model has 70 billion parameters, so it would require approximately 260.77 GB of memory (32 * 70,000,000,000 bits ≈ 260.77 GB). But that’s just the start. During training, we also need to store gradients for each parameter, which adds another 260 GB. Additionally, storing the first moment (inertia) and the second moment (adaptive learning rate) of optimizer like Adam for each parameter requires another 260 GB each.

In total, just to train a model with 70 billion parameters, you’d need approximately 1040 GB of GPU memory. And this doesn’t even account for the memory needed for activations, which are related to the batch size, data size (e.g., sequence length), and model architecture. Although we won’t include activations in our future calculations to maintain generality, it's worth noting that they occupy a comparable amount of memory to the model’s weights.

Having established the memory requirements for training in float32, let's explore how Mixed Precision works.

The key idea behind Mixed Precision is whether we can achieve sufficient accuracy by training models in float16, thereby reducing memory consumption and computation time by half. However, we can't simply convert all computations to float16 as this would lead to [numerical instability](https://arxiv.org/abs/2010.06192v1)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/W_iFlLm64CUEtm3MNzTKT.png)

**Mixed Precision** training is a technique that enables the use of float16 without causing the model training to diverge. It involves three main strategies:
- **Maintaining two copies of the weights matrix**: A “master copy” in float32 and a float16 copy. Gradient updates are calculated using the float16 matrix but applied to the float32 matrix, making the gradient update process safer.
- **Selective precision**: Different operations accumulate errors at different rates. Some operations are always safe in float16, while others are reliable only in float32. Therefore, instead of running the entire neural network in float16, some parts are run in float16 and others in float32. This mixture of data types is what gives the technique its name—"mixed precision."
- **Loss scaling**: Since float16 has a limited range, loss scaling is used to prevent underflow. However, with the advent of bfloat16 in NVIDIA GPUs starting from the Ampere series, loss scaling is no longer necessary because bfloat16 has a similar range to float32.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/vDlps5gqM3khBs3ADMDa5.png)

Now, let’s recalculate the memory requirements for training in Mixed Precision.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/dSfhohT9rwPS6xafWunP4.png)

Each parameter now requires 16 bytes, so training Llama 70B would still require approximately 1040 GB. You might wonder why the memory usage remains the same as in float32. The reason is that while we use 2 bytes (16 bits) for weights and gradients in float16, we also store a copy of the weights in float32, adding 4 bytes per parameter.

However, the significant advantage of Mixed Precision lies in computation speed—most calculations are now done in float16, which considerably speeds up the training process.

#### 1.2. PEFT (Parameter-Efficient Fine-Tuning)
PEFT is a family of methods designed to efficiently adapt large-scale models by training only a small subset of parameters. These methods significantly reduce computational costs and memory requirements while maintaining quality comparable to full fine-tuning.

One of the most popular and effective PEFT methods is [LoRa](https://arxiv.org/abs/2106.09685).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/Q0d07jIXg43H4IAEgAJJN.png)

To understand the illustration, let's delve into the fundamental observation that makes this method effective:
>A neural network contains many dense layers which perform matrix multiplication. The weight
matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al.
(2020) shows that the pre-trained language models have a low “instrisic dimension” and can still
learn efficiently despite a random projection to a smaller subspace

This means that while training for a broad, complex task, the weight matrices in a neural network have full rank, which minimizes redundancy. However, when fine-tuning this universal model for a specialized task, not all the knowledge from the original model is necessary. Therefore, only a small fraction of the parameters needs to be trained. In simpler terms, the weight matrices can be represented by smaller matrices with fewer parameters. Thus, during full fine-tuning, the weight matrices can be considered low-rank, indicating that full fine-tuning involves some degree of redundancy.

>Inspired by this, we hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation.

Given that low-rank weight matrices suffice for full fine-tuning on a downstream task, it's reasonable to assume that the gradient updates themselves can be represented by low-rank matrices. 

>For a pre-trained weight matrix \\(W_0 \in \mathbb{R}^{d\times d}\\), we constrain its update by representing the latter with a low-rank decomposition \\(W_0 + \Delta W = W_0 + BA\\), where \\(B \in \mathbb{R}^{d\times r}\\), \\(A \in \mathbb{R}^{r\times k}\\), and the rank \\(r \ll d\\). During training, \\(W_0\\) is frozen and does not receive gradient updates, while \\(A\\) and \\(B\\) contain trainable parameters. Note both \\(W_0\\) and \\(\Delta W = BA\\) are multiplied with the same input, and their respective output vectors are summed coordinate-wise. For \\(h = W_0x\\) our modified forward pass yields:
$$ h = W_0x + \Delta W x = W_0x + BAx $$

In essence, we freeze the original model, insert low-rank adapters under the relevant weight matrices, and train these adapters to simulate the updates that would normally come from gradients. With these concepts and the formulas above, you should now understand the illustration provided.

Where do the memory and computation optimizations come from? Since the baseline model is frozen, we don’t store gradients or optimizer moments for it, and we avoid unnecessary computations. Essentially, with a few caveats, we now only need to perform inference on the baseline model, which, in the case of large models, still requires significant hardware resources. However, the trainable parameters in these adapters typically constitute less than 1% of the total parameters of the original model.

#### 1.3. Quantization и QLoRa

You might wonder, "*Since fp16 works so well, can we reduce the precision of the numbers even further—to 8 bits or even 4?*" This is the essence of quantization. However, simply downcasting to 8 or 4 bits would make the computations highly unstable, especially during training.

Quantization aims to reduce memory usage with minimal loss of accuracy. While many types of quantization exist, I'll focus on the most basic and commonly used method.

Generally, quantization is applied only during inference because training with such low-precision numbers is highly unstable. However, it's possible to train adapters on top of a quantized model — a concept I'll explore later.

So, how does basic quantization work? Let's take a look at the figure:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/zI_ZTgUutnP1YW70r5D2q.png)

In simple terms, we linearly scale the parameters in the matrices from fp32 to int8 (or int4), while keeping the constants needed for reverse dequantization. This allows us to create a highly compressed model with significantly reduced memory requirements.

During inference, as the computation moves through the layers, the required parameters are dequantized, the necessary computations are performed (e.g., in fp16), and then the parameters are re-quantized before moving to the next layer.
I am simplifying things a lot, because the analysis of quantization techniques can be a separate article, but I think you have understood the main idea.

And it turns out that after such quantization, the quality of language models on inference drops quite insignificantly. This is also due to the fact that in language modeling we do not care so much about exact probabilities in predicting the next token. Basically, we just sample from the dictionary distribution predicted by the model

You might ask, "*Can this idea be applied to training?*" While training directly on a quantized model is not feasible, training adapters on top of a quantized model is possible, which is the basis of the brilliant [QLoRA](https://arxiv.org/abs/2305.14314) 

That is, there will only be a small change in the illustration about LoRa:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/eJkmp305QaBin8vtzJf77.png)

Let’s delve deeper into the QLoRa method:
First, a bit more about quantization:
>**Block-wise k-bit Quantization.** Quantization is the process of discretizing an input from a representation that holds more information to a representation with less information. It often means taking a data type with more bits and converting it to fewer bits, for example from 32-bit floats to 8-bit Integers. To ensure that the entire range of the low-bit data type is used, the input data type is commonly rescaled into the target data type range through normalization by the absolute maximum
of the input elements, which are usually structured as a tensor. For example, quantizing a 32-bit Floating Point (FP32) tensor into a Int8 tensor with range [−127, 127]:
$$ \mathbf{X}^{\text{Int8}} = \text{round}\left(\frac{127}{\text{absmax}(\mathbf{X}^{\text{FP32}})} \mathbf{X}^{\text{FP32}}\right) = \text{round}(c^{\text{FP32}} \cdot \mathbf{X}^{\text{FP32}}),$$
where c is the *quantization constant* or *quantization scale*. Dequantization is the inverse:
$$ \text{dequant}(c^{\text{FP32}}, \mathbf{X}^{\text{Int8}}) = \frac{\mathbf{X}^{\text{Int8}}}{c^{\text{FP32}}} = \mathbf{X}^{\text{FP32}} $$
The problem with this approach is that if a large magnitude value (i.e., an outlier) occurs in the input tensor, then the quantization bins—certain bit combinations—are not utilized well with few or no numbers quantized in some bins. To prevent the outlier issue, a common approach is to chunk the input tensor into blocks that are independently quantized, each with their own quantization constant c. This can be formalized as follows: We chunk the input tensor \\(\mathbf{X} \in \mathbb{R}^{b \times h}\\) into n contiguous blocks of size B by flattening the input tensor and slicing the linear segment into \\(n = (b \times h) / B\\) blocks. We quantize these blocks independently with Equation 1 to create a quantized tensor and n quantization constants \\(c_i\\)

The Block-wise method for outlier avoidance is worth noting here.

The QLoRA authors also proposed two valuable techniques:
- **Double Quantization**: Here, even the quantization constants *c* are quantized, further saving memory.
- **4-bit NormalFloat**: Leveraging the fact that pretrained neural network weights typically have a zero-centered normal distribution, this technique allows for a more informative mapping from fp32 to int4, with higher precision near zero.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/qWvY2qGfbZfOGr8T_rewn.png)

Now, let’s understand the entire QLoRA process (L1 and L2 in the formulas correspond to A and B in the figure):
> **QLoRA**. Using the components described above, we define QLORA for a single linear layer in the quantized base model with a single LoRA adapter as follows:
$$ \mathbf{Y}^{\text{BF16}} = \mathbf{X}^{\text{BF16}} \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}, \mathbf{W}^{\text{NF4}}) + \mathbf{X}^{\text{BF16}} \mathbf{L}_1^{\text{BF16}} \mathbf{L}_2^{\text{BF16}} $$
where doubleDequant(·) is defined as:
$$ \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}, \mathbf{W}^{k\text{-bit}}) = \text{dequant}(\text{dequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}), \mathbf{W}^{4\text{bit}}) = \mathbf{W}^{\text{BF16}} $$
We use NF4 for \\(\mathbf{W}\\) and FP8 for \\(c_2\\). We use a blocksize of 64 for \\(\mathbf{W}\\) for higher quantization precision and a blocksize of 256 for \\(c_2\\) to conserve memory.
>
>For parameter updates only the gradient with respect to the error for the adapters weights \\(\frac{\partial E}{\partial \mathbf{L}_i}\\) are needed, and not for 4-bit weights \\(\frac{\partial E}{\partial \mathbf{W}}\\). However, the calculation of \\(\frac{\partial E}{\partial \mathbf{L}_i}\\) entails the calculation of \\(\frac{\partial \mathbf{X}}{\partial \mathbf{W}}\\) which proceeds via first equation with dequantization from storage \\(\mathbf{W}^{\text{NF4}}\\) to computation data type \\(\mathbf{W}^{\text{BF16}}\\) to calculate the derivative \\(\frac{\partial \mathbf{X}}{\partial \mathbf{W}}\\) in BFloat16 precision.
>
>To summarize, QLORA has one storage data type (usually 4-bit NormalFloat) and a computation data type (16-bit BrainFloat). We dequantize the storage data type to the computation data type to perform the forward and backward pass, but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat.

Let’s calculate the memory requirements for a model like Llama 70B using QLoRa: 
1. Quantizing the model to 4-bit reduces the size to 65GB.
2. Adding LoRa adapters, which, let's say, occupy about 0.25% of the original model’s parameters (175 million parameters), and training these in Mixed Precision requires approximately 2.6GB (175M * 16 bytes).
3. Of course, activations for such a large model will still require significant memory, but we'll soon discuss how gradient checkpointing can help mitigate this.

In summary, we’ve reduced the memory requirement from 1040GB to 68GB! (Using LoRa without quantization, the frozen base model in 16-bit would take 260GB, plus the same 2.6GB for training adapters).

#### 1.4. Gradient Checkpointing

Get ready for some cool visualizations!

First, let's examine the computational graph for the forward and backward passes. Don't worry if it looks complex at first—there are simpler animations below that will make everything clear.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/nULvbrYGgR53_D1-fZzC_.png)

You might be wondering, what exactly are the "scary activations" we mentioned earlier? Let's break down what we aim to achieve in the learning process:
1. Our goal is to improve the model, i.e., to reduce the error.
2. We reduce this error by adjusting the model's weights.
3. To adjust the weights, we need to calculate the gradient of the error function with respect to these weights.
4. We achieve this by using the chain rule to propagate the gradient from the error back through the network to the weights during the backward pass.
5. To do this, we at least need the error value itself.
6. This means we first need to perform a forward pass—running the input data through the entire model to generate a prediction.
7. Most critically, to compute the gradients, we need to account for all the intermediate computations and the dependent terms involved in generating the prediction. These are the activations that we store in memory from the entire forward pass so we can use them during the backward pass (represented by the gray and black sections in the picture).

This process can be schematically represented as follows:

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/SwMyFsZSETbZeksAa5bCO.gif)

Now, let's think about how we can reduce memory consumption. One idea might be to recalculate each activation during the backward pass only when we need it:

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/mHYCcFuL2Bjcm774X4rVa.gif)

However, this approach leads to too many recalculations, which negatively impacts training time. This is a classic example of the Time-Memory Trade-Off in programming.
So, what's a good compromise? Here's a solution:
1. Select several "checkpoints" along the path of the forward pass and save only those.
2. During the backward pass, instead of recalculating all activations from the start, we only need to recalculate starting from the nearest checkpoint to the left.

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/yu3lvV-c2WhXmGBg-pW0n.gif)

This method significantly reduces memory consumption, though it does come at the cost of increased training time.
