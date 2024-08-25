# Optimizing LLM Training: An Overview of Techniques 👐 📚

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/0aIWkNFWweaSYFP1oAnZD.gif)

Training large language models (LLMs) requires significant computational resources and time. However, by optimizing the training process, it's possible to cut costs, speed up development, and improve the model's overall performance. This guide offers a comprehensive exploration of various optimization strategies, covering everything from basics of memory consumption to refining the training process and distributed training.

*I want to note that this article is basically a combination of the most relevant excerpts from various articles, thanks to which I was able to achieve the highest quality and reliability in the presentation of the material.*

## 0. Introduction to Data Types
Before diving into the intricacies of model training, let's briefly explore how numbers are represented in a computer and the different types of data representations available. This foundational knowledge is crucial for understanding memory consumption during model training.

### Int16/Int8/Int4
These are standard integer types. The range of values they can represent is given by \\([-2^{n-1}, 2^{n-1} - 1]\\)

A schematic representation of an Int16 bit layout can be shown as: 1 sign bit and 15 value bits.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/WY7E6uMR73aigsfcsCq8H.png)

The more bits used, the larger the range of values that can be represented.

### Float32
In Float32, the bit layout is as follows: 1 sign bit, 8 exponent bits, and 23 mantissa bits.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/KSS-oRLsPUnQ9Vypo7UZp.png)

The formula for the value is:
$$ v = (-1)^{\text{sign}} \cdot 2^{E-127} \cdot \left(1 + \sum_{i=1}^{23} b_{23-i}2^{-i}\right) $$

The key idea behind floating-point types is that more bits allocated to the exponent allow a wider range of values, while the bits allocated to the mantissa determine the precision within that range.

### Float16
The Float16 format uses 1 sign bit, 5 exponent bits, and 10 mantissa bits.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/bX17lqakEY903HrSCZF-c.png)

The main drawback of Float16 is its limited range of values, with a maximum of 65504, making it prone to overflow in activation tensors.

### Bfloat16, или brain float
Bfloat16 is a specialized data format developed by Google Brain. It can be considered an approximation of Float32. The bit layout is 1 sign bit, 8 exponent bits, and 7 mantissa bits..

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/jeGGZP2DxQfXZZuB72iRD.png)

Notice that the number of exponent bits is the same as in Float32, meaning bfloat16 can represent the same range of values, albeit with less precision. This reduces the risk of overflow in activation

Another advantage of bf16 is the ease of converting values to Float32. This is possible because of the similar bit layout. However, not all hardware currently supports this type, especially in mobile devices.

### TensorFloat32

TensorFloat32 is an interesting 19-bit data type introduced by NVidia, supported on architectures starting with NVidia Ampere (A-100). Its bit layout consists of 1 sign bit, 8 exponent bits, and 10 mantissa bits.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/ha7W9jLH-O1BvMrG5cAQf.png)

Key features:
- The number of exponent bits matches bfloat16, and therefore Float32 as well.
- The number of mantissa bits matches Float16.

This results in an unusual but highly efficient and precise data type. It delivers excellent computational performance and is suitable for model training, although it's only available on modern NVidia GPUs.

### E4M3 и E5M2
These are new 8-bit floating-point types introduced by NVidia, ARM, and Intel in the paper [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433).
The authors propose two possible 8-bit floating-point formats:
- E4M3: 1 sign bit, 4 exponent bits, and 3 mantissa bits.
- E5M2: 1 sign bit, 5 exponent bits, and 2 mantissa bits.

Experiments show that modern LLMs and image networks can be successfully trained and even inferred using these data types. We look forward to their broader adoption and hardware support. There are also more radical ideas for 4-bit floating-point formats, such as E2M1 and E3M0.

## [1. Where Did All the Memory Go?](https://arxiv.org/abs/1910.02054)

Let’s examine the memory consumption of the current training system. For example, a 1.5B parameter GPT-2 model requires 3GB (1.5B * 16bit) of memory for its weights (or parameters) in 16-bit precision, yet, it cannot be trained on a single GPU with 32GB memory using Tensorflow or PyTorch. One may wonder where all the memory goes. During model training, most of the memory is consumed by *model states*, i.e., tensors comprising of optimizer states, gradients, and parameters. Besides these model states, the rest of the memory is consumed by activations, temporary buffers and fragmented memory which we call *residual states*. We look at the memory consumption from both in details. 

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/As3IsTE-TuYbEWLHP_DoW.jpeg)

### 1.1 Model States: Optimizer States, Gradients and Parameters

Majority of the device memory is consumed by model states during training. Consider for instance, [Adam](https://arxiv.org/abs/1412.6980), one of the most popular optimizers for DL training. Adam requires storing two optimizer states, 1) the time averaged momentum and 2) variance of the gradients to compute the updates.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/qvSg5entCT4Uk4BAOGsKW.png)

Therefore, to train a model with Adam, there has to be enough memory to hold a copy of both the momentum and variance of the gradients. In addition, there needs to be enough memory to store the gradients and the weights themselves. Of these three types of the parameter-related tensors, the optimizer states usually consume the most memory, specially when mixed-precision training is applied.

**Mixed-Precision Training** The state-of-the-art approach to train large models on the current generation of NVIDIA GPUs is via [mixed precision training](https://arxiv.org/abs/1710.03740), where parameters and activations are stored as fp16, enabling the use of the high throughput tensor core units on these GPUs. During mixed-precision training, both the forward and backward propagation are performed using fp16 weights and activations. However, to effectively compute and apply the updates at the end of the backward propagation, the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other otimizer states.

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/QmMbZaLmppCKaIo0fWtHT.gif)

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/h0p-IciIv8sVY1I3l_wUL.png)

Let’s take Adam as a concrete example. Mixed precision training of a model with Φ parameters using Adam requires enough memory to hold an fp16 copy of the parameters and the gradients, with memory requirements of 2Φ and 2Φ bytes respectively. In addition, it needs to hold the optimizer states: an fp32 copy of the parameters, momentum and variance, with memory requirements of 4Φ, 4Φ, and 4Φ bytes, respectively.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/zlD-T_HtyeSKEY_zhqLix.png)

In total, this results 16Φ bytes of memory requirement. For a model such as GPT-2 with 1.5 Billion parameters, this leads to a memory requirement of at least 24 GB, which is significantly higher than the meager 3 GB of memory required to hold the fp16 parameters alone.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/3DuZdDRbhLK46MfVhKJJX.png)

### 1.2 Residual Memory Consumption
**Activations** can take up a significant amount of memory during training. As a concrete example, the 1.5B parameter GPT-2 model trained with sequence length of 1K and batch size of 32 requires about 60 GB of memory. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/iuytzmuBVVrIPUb72hj3s.png)

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/GgNu3RrWs0ls9AX3jFUmk.gif)

The activation memory of a transformer-based model is proportional to the number of *transformer layers* × *hidden dimensions* × *sequence length* × *batch size*. 

[**Activation checkpointing**](https://arxiv.org/abs/1604.06174) (or gradient checkpointing) is a common approach to reduce the activation memory by approximately the square root of the total activations at the expense of 33% re-computation overhead. This would reduce the activation memory consumption of this model from 60 GB to about 8 GB. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/BsRo4b2J31zUFr-KMcs_n.png)

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/ThPtDpjoHZ0GLBsRBfxAR.gif)

Despite the significant reduction, the activation memory can grow quite large for bigger models even with activation checkpointing. For example, a GPT-like model with 100 billion parameters requires around 60 GB of memory for batch size 32, even when using activation checkpointing.

**Temporary buffers** used for storing intermediate results consumes non-trivial amount of memory for large models. Operations such as gradient all-reduce, or gradient norm computation tend to fuse all the gradients into a single flattened buffer before applying the operation in an effort to improve throughput. For example, the bandwidth of all-reduce across devices improves with large message sizes. While the gradient themselves are usually stored as fp16 tensors, the fused buffer can be an fp32 tensor depending on the operation. When the size of the model is large, these temporary buffer sizes are non-trivial. For example, for a model with 1.5B parameters, a flattened fp32 buffer would required 6 GB of memory

**Memory Fragmentation**: So far we have discussed the actual memory consumption during training. Additionally, it is possible to run out of usable memory even when there is plenty of available memory. This can happen with memory fragmentation. A request for a memory will fail if there isn’t enough contiguous memory to satisfy it, even if the total available memory is larger than requested. We observe significant memory fragmentation when training very large models, resulting in out of memory issue with over 30% of memory still available in some extreme cases.

## 2. Quantization
Quantization in deep learning is the process of reducing the precision of the numbers used to represent a model's parameters (weights) and computations, typically from 32-bit floating-point (FP32) to lower bit-width formats like 16-bit floating-point (FP16), 8-bit integers (INT8), or even lower. The main goal of quantization is to decrease the model's size, reduce memory usage, and accelerate inference by enabling the model to run efficiently on hardware with limited computational resources.

In general, it is not possible to perform pure 4bit/8bit training on quantized models. However, you can train these models by leveraging parameter efficient fine tuning methods (PEFT) and train for example adapters on top of them. We'll dive into this approach in the next section

The simplest form of "quantization" is to convert parameters from fp32 to fp16. During training, the main weights are always stored in FP32, but in practice, the half-precision weights often provide similar quality during inference as their fp32 counterpart - a precise reference of the model is only needed when it receives multiple gradient updates. This means we can use the half-precision weights and use half the GPUs to accomplish the same outcome.

It'd be amazing to cut precision further, but the inference quality outcome starts to drop dramatically at lower precision. That's why we need trickier ways to do it.

Quantization is done by essentially “rounding” from one data type to another. For example, if one data type has the range 0..9 and another 0..4, then the value “4” in the first data type would be rounded to “2” in the second data type. However, if we have the value “3” in the first data type, it lies between 1 and 2 of the second data type, then we would usually round to “2”. This shows that both values “4” and “3” of the first data type have the same value “2” in the second data type. This highlights that quantization is a noisy process that can lead to information loss, a sort of lossy compression.

### 2.1 Asymmetric and Symmetric Linear Quantization
Let’s start with the illustrations:

**Asymmetric**:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/E1qaUh4uRmMXMfmxPlSiu.png)

**Symmetric**:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/QMMum7lBhmZlPj-BCANn8.png)

In essence, we're mapping a continuous range of real numbers into an integer range. The process can be visualized as follows:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/JwT-g6-J31Hce_xylvAxD.png)

Here, **S** and **Z** are the quantization parameters, calculated during the quantization process. **S** (scale) determines the transformation's scale, and **Z** (zero point) corresponds to the zero value in the quantized domain.
- **Asymmetric**
  - \\(S = \frac {r_{max}-r_ {min}}{q_{max}-q_{min}} \\)
  - \\(Z = \left[q_{min} - \frac{r_{min}}{S}\right]\\)
  - \\(X_{quantized} = \left[\frac{X}{S} + Z\right]\\)
  - \\(X_{dequantized} = S(X_{quantized} - Z)\\)

- **Symmetric**
  - The quantization range is determined by the maximum absolute value of the data.
  - \\(S = \frac{|r|_{max}}{2^{N-1} - 1} \\)
  - \\(Z = 0\\)
  - \\(X_{quantized} = \left[\frac{X}{S}\right]\\)
  - \\(X_{dequantized} = SX_{quantized}\\)
  - To maintain symmetry, one value is typically removed from the quantized data type. For example, the signed int8 range of [-128, 127] becomes [-127, 127].

where \\( [*] \\) denotes rounding.

The advantage of asymmetric quantization is its ability to better handle asymmetric data distributions, whereas symmetric quantization benefits from simplicity and speed. With symmetric quantization, there's no need to store a zero-point, and dequantization is a simple multiplication by a constant.

Example of Symmetric quantization:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/6qS8qC8WTNdZFcnV16ufS.png)

The result is an 8-bit integer tensor with a quantization constant of 23.5. This allows for reduced storage requirements and, if necessary, conversion back to the original 32-bit floating-point representation, albeit with some loss of precision.

### 2.2 What to Quantize?

The standard approach is to quantize the model's weights. This requires no additional manipulations—just apply the formulas.

You can also quantize the outputs of layers, known as activations. To do this, you need to estimate the range of values that appear in activation tensors. This is done by running data from the training dataset through the trained neural network and collecting statistics. Using this information, you determine the quantization parameters. This method is called static quantization.

In dynamic quantization, activations are quantized during inference. This approach can yield better quality, but it introduces challenges: finding the quantization parameters dynamically during inference makes the method more complex and computationally expensive, though it ensures the parameters are always up-to-date.

### 2.3 When to Quantize?

Preparing a network for quantization can be done during training, known as **Quantize-Aware Training**. In this approach, special blocks are embedded in the neural network, and quantized inference is simulated during training.

Quantize-Aware Training is complex and requires more computational resources, but it produces a model that is "adapted" to working with quantized values, potentially offering higher accuracy.

In the **Post Training Quantization** method, an already trained model is quantized. For activation quantization, you pass data from a calibration dataset through the trained network, collect tensor statistics, and then quantize. If you’re only quantizing weights, no additional data is needed since all necessary information is already in the tensors. This method is simpler and faster than Quantize-Aware Training but is typically less accurate.

### 2.4 Granularity

Quantization can be applied with varying levels of granularity. The most basic approach is to quantize the entire network at once, resulting in a single scale factor S for the entire model. This often leads to unsatisfactory results.

A better approach is to quantize tensors individually, allowing each tensor to have its own scale factor. You can go even further and quantize rows or columns within each tensor, giving each row (or column) its own scale factor. Although this increases the storage requirements for scale factors, it significantly improves the accuracy of computations.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/xbY-NhjLaCY88RPi5PNee.png)

You can also divide the tensor into smaller blocks, which yields even greater accuracy. This approach helps mitigate the impact of outliers in matrices, a topic we'll explore further.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/S1SvT4tE5OEVvTskumu3c.png)

In summary, the smaller the granularity, the fewer scale factors you need to store; conversely, the higher the granularity, the closer the quantized computations are to the original.

### 2.5 Data Types

Quantized neural network models typically involve two types of data:

- **Quantized type** —  the type used to store tensors.
- **Computation type** — the type used for performing calculations.

Unfortunately, these two types don't always match. For example, your hardware might not support operations in a specific quantized type. Efficient matrix multiplication kernels for certain quantized types may not exist. In such cases, you’ll need to convert the matrix to a computation type before performing calculations. The computation type also helps avoid overflow issues in activations since multiplying 8-bit numbers can easily exceed the data type's limits.

### 2.6 The Problem of Outliers
Consider the example of symmetric quantization:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/AHIUqmayD-GX7PpEPirpg.png)

What happens if an outlier is present in the input tensor?

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/M-NJYW1SuSDNU4qiHa5Dv.png)

The weights get "compressed" into a narrow range, becoming indistinguishable. The model's quality is compromised. In this case, a single outlier ruined the entire matrix.

As the number of parameters increases, standard quantization techniques, which we discussed above, begin to fail. When the number of parameters exceeds 6.7 billion, quantized models [lose significant quality](https://arxiv.org/abs/2208.07339). This occurs due to the increasing number of outliers in the matrices.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/juzSha_nIdR3znammiwgO.png)

### 2.7 LLM.int8()

The authors of the [paper](https://arxiv.org/abs/2208.07339) introduced a method to quantize large models (up to 175 billion parameters) from the usual 16- or 32-bit floating-point weights to 8-bit integers with minimal loss in quality. The key idea is to handle outliers separately, as they constitute a very small portion of the data (0.1–1% of all values) and are concentrated in specific channels of the activation tensors.

Let's consider the multiplication of the activation matrix **𝑋** by the weight matrix **𝑊**. The columns of **𝑋** are divided into two groups: those containing at least one outlier and those without any. This division results in two new weight matrices derived from the original **𝑊**.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/bHQ1P230U88BMC6cUh_yK.png)

It's important to note that the i-th column of activations **𝑋** interacts only with the i-th row of weights **𝑊**. Hence, the matrix **𝑊** can also be split into two parts by separating the rows corresponding to the outlier columns of **𝑋**.

As a result, we obtain two groups of matrices: one with outliers and one without. Each group is then multiplied separately, and the results are summed. This sum is equivalent to the usual matrix multiplication.

Most of the values will fall into matrices without outliers, which can be easily quantized to 8 bits, allowing for efficient operations. The matrices containing outliers are left in their original 16-bit type to ensure computations remain accurate.

However, the increased quantization accuracy comes at the cost of reduced performance due to the overhead of additional computations. The authors' benchmarks show a 15–23% decrease in inference speed on BLOOM-176B compared to the 16-bit default. 

### 2.8 GPTQ

Quantization is rapidly evolving, with increasingly [new and efficient approaches](https://huggingface.co/docs/transformers/quantization/overview) emerging. We won’t delve further into this topic but will briefly explore one more alternative approach.

Let's reconsider the problem: Is rounding to the nearest integer the optimal solution? Perhaps not. Our actual goal is to find a quantized weight matrix \\(\hat{W}\\) that, when multiplied by the activation matrix, produces a result as close as possible to the original:
$$ \min_{\hat{W}} \|XW - X\hat{W}\|_2^2 $$

This involves a lot of mathematics and engineering solutions, but the idea should be clear. For more details, you can refer to the [original paper](https://arxiv.org/abs/2210.17323)

It's important to note that everything discussed so far has focused solely on using quantized models for inference optimization. But what about training?

## 3. PEFT (Parameter-Efficient Fine-Tuning), LoRA and QLoRa

[PEFT](https://huggingface.co/docs/peft/index) is a family of methods designed to efficiently adapt large-scale models by training only a small subset of parameters. These methods significantly reduce computational costs and memory requirements while maintaining quality comparable to full fine-tuning.

### 3.1 LoRA: Low-Rank Adaptation
One of the most popular and effective PEFT methods is [LoRa](https://arxiv.org/abs/2106.09685).

![image/gif](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/zqXpRb2muYPpSGKuYc6IK.gif)

To understand the illustration, let's delve into the fundamental observation that makes this method effective:
>A neural network contains many dense layers which perform matrix multiplication. The weight
matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al.
(2020) shows that the pre-trained language models have a low “instrisic dimension” and can still
learn efficiently despite a random projection to a smaller subspace

This means that while training for a broad, complex task, the weight matrices in a neural network have full rank, which minimizes redundancy. However, when fine-tuning this universal model for a specialized task, not all the knowledge from the original model is necessary. Therefore, only a small fraction of the parameters needs to be trained. In simpler terms, the weight matrices can be represented by smaller matrices with fewer parameters. Thus, during full fine-tuning, the weight matrices can be considered low-rank, indicating that full fine-tuning involves some degree of redundancy.

>Inspired by this, we hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation.

Given that low-rank weight matrices suffice for full fine-tuning on a downstream task, it's reasonable to assume that the gradient updates themselves can be represented by low-rank matrices. 

>For a pre-trained weight matrix \\(W_0 \in \mathbb{R}^{d\times d}\\), we constrain its update by representing the latter with a low-rank decomposition \\(W_0 + \Delta W = W_0 + BA\\), where \\(B \in \mathbb{R}^{d\times r}\\), \\(A \in \mathbb{R}^{r\times d}\\), and the rank \\(r \ll d\\). During training, \\(W_0\\) is frozen and does not receive gradient updates, while \\(A\\) and \\(B\\) contain trainable parameters. Note both \\(W_0\\) and \\(\Delta W = BA\\) are multiplied with the same input, and their respective output vectors are summed coordinate-wise. For \\(h = W_0x\\) our modified forward pass yields:
$$ h = W_0x + \Delta W x = W_0x + BAx $$

In essence, we freeze the original model, insert low-rank adapters under the relevant weight matrices, and train these adapters to simulate the updates that would normally come from gradients. With these concepts and the formulas above, you should now understand the illustration provided.

>The most significant benefit comes from the reduction in memory and storage usage. For a large Transformer trained with Adam, we reduce that VRAM usage by up to 2/3 if \\(r \ll d\\) as we do not need to store the gradients and optimizer states for the frozen parameters. We also observe a 25% speedup during training on GPT-3 175B compared to full fine-tuning as we do not need to calculate the gradient for the vast majority of the parameters.

### 3.2 QLoRA

[QLoRA](https://arxiv.org/abs/2305.14314) uses 4-bit quantization to compress a pretrained language model. The LM parameters are then frozen and a relatively small number of trainable parameters are added to the model in the form of Low-Rank Adapters. During finetuning, QLoRA backpropagates gradients through the frozen 4-bit quantized pretrained language model into the Low-Rank Adapters. The LoRA layers are the only parameters being updated during training.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/eJkmp305QaBin8vtzJf77.png)

QLoRA has one storage data type (usually 4-bit NormalFloat) for the base model weights and a computation data type (16-bit BrainFloat) used to perform computations. QLoRA dequantizes weights from the storage data type to the computation data type to perform the forward and backward passes, but only computes weight gradients for the LoRA parameters which use 16-bit bfloat. The weights are decompressed only when they are needed, therefore the memory usage stays low during training and inference.

Let’s dive into this in more detail. First, let's look at the quantization method used by the authors of the paper. As we remember from the previous section, there are many different approaches.

>**Block-wise k-bit Quantization.** Quantization is the process of discretizing an input from a representation that holds more information to a representation with less information. It often means taking a data type with more bits and converting it to fewer bits, for example from 32-bit floats to 8-bit Integers. To ensure that the entire range of the low-bit data type is used, the input data type is commonly rescaled into the target data type range through normalization by the absolute maximum
of the input elements, which are usually structured as a tensor. For example, quantizing a 32-bit Floating Point (FP32) tensor into a Int8 tensor with range [−127, 127]:
$$ \mathbf{X}^{\text{Int8}} = \text{round}\left(\frac{127}{\text{absmax}(\mathbf{X}^{\text{FP32}})} \mathbf{X}^{\text{FP32}}\right) = \text{round}(c^{\text{FP32}} \cdot \mathbf{X}^{\text{FP32}}),$$
where c is the *quantization constant* or *quantization scale*. Dequantization is the inverse:
$$ \text{dequant}(c^{\text{FP32}}, \mathbf{X}^{\text{Int8}}) = \frac{\mathbf{X}^{\text{Int8}}}{c^{\text{FP32}}} = \mathbf{X}^{\text{FP32}} $$
The problem with this approach is that if a large magnitude value (i.e., an outlier) occurs in the input tensor, then the quantization bins—certain bit combinations—are not utilized well with few or no numbers quantized in some bins. To prevent the outlier issue, a common approach is to chunk the input tensor into blocks that are independently quantized, each with their own quantization constant c. This can be formalized as follows: We chunk the input tensor \\(\mathbf{X} \in \mathbb{R}^{b \times h}\\) into n contiguous blocks of size B by flattening the input tensor and slicing the linear segment into \\(n = (b \times h) / B\\) blocks. We quantize these blocks independently with Equation 1 to create a quantized tensor and n quantization constants \\(c_i\\)

As we can see, the authors address the important issue of outliers, which we discussed earlier, by breaking down matrices into many small blocks, thereby minimizing the potential variance within a single quantization block.

Additionally, to fully understand how QLoRA works, we need to consider two more important concepts.

>**Double Quantization**. We introduce Double Quantization (DQ), the process of quantizing the quantization constants for additional memory savings. While a small blocksize is required for precise 4-bit quantization (because of outliers), it also has a considerable memory overhead. For example, using 32-bit constants and a blocksize of 64 for **W**, quantization constants add 32/64 = 0.5 bits per parameter on average. Double Quantization helps reduce the memory footprint of quantization constants.

This reduces the memory footprint per parameter from 0.5 bits to 0.127 bits

**Normal Float 4 (NF4)**  Leveraging the fact that pretrained neural network weights typically have a zero-centered normal distribution, this technique allows for a more informative mapping from fp32 to int4, aking into account the increased density near 0.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/-4TEpbhrA-bwfI95ASG4_.png)

Now, we are ready to understand the entire QLoRA process (L1 and L2 in the formulas correspond to B and A):
> **QLoRA**. Using the components described above, we define QLORA for a single linear layer in the quantized base model with a single LoRA adapter as follows:
$$ \mathbf{Y}^{\text{BF16}} = \mathbf{X}^{\text{BF16}} \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}, \mathbf{W}^{\text{NF4}}) + \mathbf{X}^{\text{BF16}} \mathbf{L}_1^{\text{BF16}} \mathbf{L}_2^{\text{BF16}} $$
where doubleDequant(·) is defined as:
$$ \text{doubleDequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}, \mathbf{W}^{k\text{-bit}}) = \text{dequant}(\text{dequant}(c_1^{\text{FP32}}, c_2^{k\text{-bit}}), \mathbf{W}^{4\text{bit}}) = \mathbf{W}^{\text{BF16}} $$
We use NF4 for \\(\mathbf{W}\\) and FP8 for \\(c_2\\). We use a blocksize of 64 for \\(\mathbf{W}\\) for higher quantization precision and a blocksize of 256 for \\(c_2\\) to conserve memory.
>
>For parameter updates only the gradient with respect to the error for the adapters weights \\(\frac{\partial E}{\partial \mathbf{L}_i}\\) are needed, and not for 4-bit weights \\(\frac{\partial E}{\partial \mathbf{W}}\\). However, the calculation of \\(\frac{\partial E}{\partial \mathbf{L}_i}\\) entails the calculation of \\(\frac{\partial \mathbf{X}}{\partial \mathbf{W}}\\) which proceeds via first equation with dequantization from storage \\(\mathbf{W}^{\text{NF4}}\\) to computation data type \\(\mathbf{W}^{\text{BF16}}\\) to calculate the derivative \\(\frac{\partial \mathbf{X}}{\partial \mathbf{W}}\\) in BFloat16 precision.
>
>To summarize, QLORA has one storage data type (usually 4-bit NormalFloat) and a computation data type (16-bit BrainFloat). We dequantize the storage data type to the computation data type to perform the forward and backward pass, but we only compute weight gradients for the LoRA parameters which use 16-bit BrainFloat.

QLORA reduces the average memory requirements of finetuning a 65B parameter model from >780GB of GPU memory to <48GB without degrading the runtime or predictive performance compared to a 16-bit fully finetuned baseline. This marks a significant shift in accessibility of LLM finetuning: now the largest publicly available models to date finetunable on a single GPU.

## 4. Additional techniques

### 4.1 Flash Attention

Scaling the transformer architecture is heavily bottlenecked by the self-attention mechanism, which has quadratic time and memory complexity. Recent developments in accelerator hardware mainly focus on enhancing compute capacities and not memory and transferring data between hardware. This results in attention operation having a memory bottleneck.

Standard attention mechanism uses High Bandwidth Memory (HBM) to store, read and write keys, queries and values. HBM is large in memory, but slow in processing, meanwhile SRAM is smaller in memory, but faster in operations. In the standard attention implementation, the cost of loading and writing keys, queries, and values from HBM is high. It loads keys, queries, and values from HBM to GPU on-chip SRAM, performs a single step of the attention mechanism, writes it back to HBM, and repeats this for every single attention step.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/g4EnH54JxS_ZFvNgtVNce.png)

[**FlashAttention**](https://tridao.me/publications/flash3/flash3.pdf) is an algorithm that reorders the attention computation and leverages tiling and recomputation to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. It uses tiling to load blocks of inputs from HBM (GPU memory) to SRAM (fast cache), perform attention with respect to that block, and update the output in HBM. By not writing the large intermediate attention matrices to HBM, we reduce the amount of memory reads/writes, which brings 2-4x wallclock time speedup.

Diagram of FlashAttention forward pass: with tiling and softmax rescaling, we operate by blocks and avoid having to read/write from HBM, while obtaining the correct output with no approximation.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/_Xf7ZPpoX6o-17ARq6B4E.png)

For FP16:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/-VbVssfm8mkToIrFxJgWk.png)

### 4.2. Gradient Accumulation

**Gradient accumulation** is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory. This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed.

For instance, if the gradient accumulation factor is set to 2, the process works as follows: We first calculate the gradient on one batch, which gives us a direction on the [loss function landscape](https://losslandscape.com/). Instead of updating the model weights immediately, we calculate another gradient from the next batch, obtaining a potentially different direction. By adding these two gradients together, we find a more accurate path in the loss landscape. To ensure the final update step is properly scaled, we divide the accumulated gradient by the number of batches, preventing any artificial inflation of the step size.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/UJ2nzxFp7EUquy61gDnBZ.png)

This technique is particularly useful when only small batch sizes can fit into memory, which might otherwise lead to overly noisy updates and less stable training.

### 4.3 8-bit optimizers

Stateful optimizers maintain gradient statistics over time, for example, the exponentially smoothed sum (SGD with momentum) or squared sum (Adam) of past gradient values.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/ugAlqFWuFGi3tlLLY7nMq.png)

This state can be used to accelerate optimization compared to plain stochastic gradient descent, but uses memory that might otherwise be allocated to model parameters. As a result, this limits the maximum size of models that can be trained in practice. Now take a look at the biggest models that can be trained with [8-bit optimizers](https://arxiv.org/abs/2110.02861).

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/jbP32Szbir8_Wmj2zqZVU.png)

The idea, as you might have guessed, is to quantize the optimizer states to 8-bit.

To overcome the resulting computational, quantization and stability challenges, 8-bit optimizers have three components:
1. **Block-wise quantization**: divides input tensors into smaller blocks that are independently quantized, isolating outliers and distributing the error more equally over all bits. Each block is processed in parallel across cores, yielding faster optimization and high precision quantization.
2. **Dynamic quantization**: quantizes both small and large values with high precision.
3. **Stable embedding layer**: improves stability during optimization for models with word embeddings.

With these components, performing an optimizer update with 8-bit states is straightforward. The 8-bit optimizer states are dequantized to 32-bit before you perform the update, and then the states are quantized back to 8-bit for storage.

The 8-bit to 32-bit conversion happens element-by-element in registers, meaning no slow copies to GPU memory or additional temporary memory are needed to perform quantization and dequantization. For GPUs, this makes 8-bit optimizers much faster than regular 32-bit optimizers.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/GQaxPFty1nO2JDZmuWeAz.png)

### 4.4 Sequence Packing

When finetuning a large language model with either full-parameter or parameter-efficient finetuning, GPU underutilization is a common problem due to an inefficient data pipeline. This is because most finetuning datasets have a skewed distribution of sequence lengths, with many short sequences and a few long sequences, following Zipf’s Law.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/tqWfrWvdY1B_zOJgJGAgI.png)

Transformer models can only take in fixed length inputs, so the input has to be padded with many unused pad tokens, which is inefficient in two ways:
- Computation performed on the pad values is eventually ignored for model output, resulting in wasted FLOPs.
- Micro batch size is often limited by the batch which contains longer sequences, so that most other micro batches have underutilized GPU memory.

**Sequence packing** is a training technique where multiple training sequences (examples) are concatenated together into one long sequence (pack). This eliminates the need for padding and allows more tokens to be processed in each micro batch, maximizing both GPU compute and GPU memory.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/2ZH5pNV9cQ7UfAb0FIGxB.png)

While sequences for pretraining can be concatenated naively, this is not the case for SFT and instruction fine-tuning where each input sequence should be treated individually. The conventional solution is to build an extended attention mask to mark the sequence id each token belongs to, and mask out attention values between sequences.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/gGOyIt3K06cbKJwkgvW1A.png)

However, this increases the complexity of attention from \\(\left(\sum_i s_i^2\right)\\) to \\(\left(\sum_i s_i\right)^2\\) where \\(s_i\\) is the length of the ith subsequence. In practice, the conventional solution puts a limit on the length of packing.

Instead, [NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html) provides a highly optimized version of sequence packing which makes use of variable-length attention kernels in FlashAttention and TransformerEngine. With this, attention values between sequences are never calculated, so the complexity of attention remains at \\(\left(\sum_i s_i^2\right)\\). This allows packing sequences to arbitrary lengths so that GPU memory can be fully utilized.

All things considered, NeMo’s implementation of sequence packing provides *(on Llama 7B with Dolly dataset)*:
- Up to 10X performance improvement in terms of FLOPs
- Up to 6X performance improvement in terms of training time
- No impact on model convergence

### 4.5 torch.compile()
[**torch.compile**](https://pytorch.org/get-started/pytorch-2.0/#pytorch-2x-faster-more-pythonic-and-as-dynamic-as-ever) makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, all while requiring minimal code changes. 

Whenever you wrap your model under torch.compile, the model goes through the following steps before execution:
1. **Graph Acquisition**: The model is broken down and re-written into subgraphs. Subgraphs that can be compiled/optimized are flattened, whereas other subgraphs which can’t be compiled fall back to the eager model.
2. **Graph Lowering**: All PyTorch operations are decomposed into their chosen backend-specific kernels.
3. **Graph Compilation**: All the backend kernels call their corresponding low-level device operations.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/D4BlLDnfvNUpvVfdFCd6z.png)

On 163 open source models from different libraries (e.g., TIMM, TorchBench, and Hugging Face), torch.compile provided 30%-200% speedups on NVIDIA A100s.

### 4.6 Multi-query Attention (MQA) and Grouped-query Attention (GQA)
[**Multi-query Attention (MQA)**](https://arxiv.org/abs/1911.02150) and [**Grouped-query Attention (GQA)**](https://arxiv.org/abs/2305.13245) are modifications of the traditional multihead attention mechanism in Transformer models. These methods improve the efficiency and effectiveness of attention mechanisms.

- **MQA** treats all attention heads as a single group, reducing computational complexity and accelerating training times. It is beneficial when model scalability or limited computational resources are concerns.
- **GQA** groups the heads into clusters, each processing a subset of queries independently. This method balances the detailed focus of traditional multihead attention with the broad approach of MQA, enhancing nuanced input data processing.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/7DrJrVJdFxpwXhvFRuwW0.png)

These attention variants offer:
- **Reduced computational load**: Both methods decrease computation, beneficial for large models.
- **Increased processing speed**: Simplifying attention leads to faster training and inference.

## 5. Collective Operations
Before diving into distributed training, it’s beneficial to first understand the basic operations involved in multi-GPU and multi-node communication.

For this purpose, we'll focus on the [NVIDIA NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)

The NVIDIA Collective Communication Library (NCCL) implements multi-GPU and multi-node communication primitives optimized for NVIDIA GPUs and Networking. NCCL provides routines such as all-gather, all-reduce, broadcast, reduce, reduce-scatter as well as point-to-point send and receive that are optimized to achieve high bandwidth and low latency over PCIe and NVLink high-speed interconnects within a node and over NVIDIA Mellanox Network across nodes.

Leading deep learning frameworks such as Caffe2, Chainer, MxNet, PyTorch and TensorFlow have integrated NCCL to accelerate deep learning training on multi-GPU multi-node systems.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/7FXIGEEFaWJaLN_Ttydn0.png)

Collective operations have to be called for each rank (hence CUDA device) to form a complete collective operation. Failure to do so will result in other ranks waiting indefinitely.

### 5.1 AllReduce
The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank.

In a sum allreduce operation between k ranks, each rank will provide an array in of N values, and receive identical results in array out of N values, where out[i] = in0[i]+in1[i]+…+in(k-1)[i].

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/fEKFr4jK-NCawJmD4ZbPq.png)

### 5.2 Broadcast
The Broadcast operation copies an N-element buffer from the root rank to all the ranks.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/PSZn_fdIu5oAiQhx2us4X.png)

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

### 5.3 Reduce
The Reduce operation performs the same operation as AllReduce, but stores the result only in the receive buffer of a specified root rank.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/tXisKmL7_Q9MqEjJA2twc.png)

Important note: The root argument is one of the ranks (not a device number), and is therefore impacted by a different rank to device mapping.

Note: A Reduce, followed by a Broadcast, is equivalent to the AllReduce operation.

### 5.4 ReduceScatter
The ReduceScatter operation performs the same operation as Reduce, except that the result is scattered in equal-sized blocks between ranks, each rank getting a chunk of data based on its rank index.

The ReduceScatter operation is impacted by a different rank to device mapping since the ranks determine the data layout.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/Z6wffjJ3IPcpyyfpyzcHD.png)

### 5.5 AllGather
The AllGather operation gathers N values from k ranks into an output buffer of size k*N, and distributes that result to all ranks.

The output is ordered by the rank index. The AllGather operation is therefore impacted by a different rank to device mapping.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/fvBzt8G7NsjKQTDvJanDK.png)

Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

## 6. Distributed Training
Principally, there are two approaches to parallelism — data parallelism and model parallelism.

### 6.1 DP - Data Parallelism
Parallelization is a key strategy on training large models at scale. For a model that fits in the device memory for training, data parallelism (DP) is used to scale training to multiple devices. In DP, model parameters are replicated on each device. At each step, a mini-batch is divided evenly across all the data parallel processes, such that each process executes the forward and backward propagation on a different subset of data samples, and uses averaged gradients across processes to update the model locally.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/uZ60Sbc6Q7ZrvzQSIR1xX.png)

### 6.2 Model Parallelism, Tensor Parallelism, Pipeline Parallelism
When a model does not fit in the device memory, model parallelism split the model among processes, in vertical or horizontal way.

#### 6.2.1 Naive Model Parallelism 
This approach involves distributing groups of model layers across multiple GPUs by assigning specific layers to specific GPUs. As data flows through these layers, it is moved to the same GPU as the layer, while the other layers remain untouched.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/MrYIdXnXDTQNGbL3_8DMC.png)

In this example, when data moves through layers within one GPU, it’s no different from regular forward pass. However, moving data between layers on different GPUs results in a communication overhead. If the participating GPUs are on the same compute node (e.g. same physical machine) this copying is fast, but if the GPUs are distributed across different compute nodes (e.g. multiple machines), the communication overhead could be substantially greater.

The main problem with Naive Model Parallelism is that **аll but one GPU are idle at any given moment**, which is very inefficient.

#### 6.2.2 Pipeline Parallelism 
PP is almost identical to a naive MP, but it solves the GPU idling problem by chunking the incoming batch into micro-batches and artificially creating a pipeline, which allows different GPUs to concurrently participate in the computation process.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/8B1qNEhiSN8EcAb_jgxlo.png)

But this comes at the expense of a great deal of technical complication.

#### 6.2.3 Tensor Parallelism 
In Tensor Parallelism, each GPU processes a slice of a tensor and only aggregates the full tensor for operations requiring it. So, unlike Model Parallelism (MP), we don't have to wait for the previous GPUs to finish processing the previous layers of the model. This allows for more efficient processing and reduced idle time.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/T3qIAk8Dba9gM_zZyOV0W.png)

The main building block of any transformer is a fully connected **nn.Linear** followed by a nonlinear activation **GeLU**. The dot dot-product part of it, following the [Megatron’s paper](https://arxiv.org/abs/2104.04473) notation, can be written as **Y = GeLU(XA)**, where **X** is an input vector, **Y** is the output vector, and **A** is the weight matrix.

If we look at the computation in matrix form, you can see how the matrix multiplication can be split between multiple GPUs:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/oOGpgNvibHgLg7ST7jyH1.png)

If we split the weight matrix **A** column-wise across **N** GPUs and perform matrix multiplications **XA_1** through **XA_n** in parallel, then we will end up with **N** output vectors **Y_1, Y_2, ..., Y_n** which can be fed into **GeLU** independently:
$$ [Y_1, Y_2] = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)] $$

Using this principle, we can update a multi-layer perceptron of arbitrary depth, without the need for any synchronization between GPUs until the very end, where we need to reconstruct the output vector from shards. The Megatron-LM paper authors provide a helpful illustration for that:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/k6xXMiY3m_o3s-81LXvnK.png)

Parallelizing the multi-headed attention layers is even simpler, since they are already inherently parallel, due to having multiple independent heads!

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/nqdbDwvP_kVt3_wQloYzX.png)

### 6.4 FSDP - Fully Sharded Data Parallel
In DataParallel training, each process/ worker owns a replica of the model and processes a batch of data, finally it uses all-reduce to sum up gradients over different workers. In DP the model weights and optimizer states are replicated across all workers. [FSDP](https://arxiv.org/abs/2304.11277) is a type of data parallelism that shards model parameters, optimizer states and gradients across DP ranks.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/CC79O1a0bWmQ7qNBxBvjv.png)

Usually, model layers are wrapped with FSDP in a nested way, so that only layers in a single FSDP instance need to gather the full parameters to a single device during forward or backward computations. The gathered full parameters will be freed immediately after computation, and the freed memory can be used for the next layer’s computation. In this way, peak GPU memory could be saved and thus training can be scaled to use a larger model size or larger batch size. To further maximize memory efficiency, FSDP can offload the parameters, gradients and optimizer states to CPUs when the instance is not active in the computation.

#### 6.4.1 FSDP Units

#### 6.4.2 Sharding Strategy

#### 6.4.3 FSDP Workflow
**In constructor**:
- Shard model parameters and each rank only keeps its own shard

**Forward pass**:
1. Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
2. Run forward computation
3. Discard parameter shards it has just collected

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/7zvopsLHueGXc0FmKGi28.png)

**Backward pass**:
1. Run all_gather to collect all shards from all ranks to recover the full parameter in this FSDP unit
2. Run backward computation
3. Run reduce_scatter to sync gradients
4. Discard parameters.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/IAfgPwAKNXVG5Y84ReqAl.png)
