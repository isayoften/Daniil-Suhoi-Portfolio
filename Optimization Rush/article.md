# Optimizing LLM Training: An Overview of Techniques 👐 📚

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/m9v01CkHNjLKvt1eUHTWz.png)

Training large language models (LLMs) requires significant computational resources and time. However, by optimizing the training process, it's possible to cut costs, speed up development, and improve the model's overall performance. This guide offers a detailed exploration of various optimization strategies, covering everything from choosing the right model to refining the learning process.

## 0. Небольшое введение в типы данных
Давайте для начала вкратце разберем, как числа представляются в компьютере и какие разновидности данного представления существуют. Нам это очень сильно понадобится в дальнейшем для понимания потребления памяти во время обучения моделей.

### Int16/Int8/Int4
Самые обыкновенные целочисленные типы. Диапазон значений - \\([-2^{n-1}, 2^{n-1} - 1]\\)

Схематично битовое представление Int16 можно показать так: 1 бит знака и 15 бит на значение.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/WY7E6uMR73aigsfcsCq8H.png)

Чем больше битов, тем точнее можно представить диапазон значений.

### Float32
Здесь битовое представление выглядит так: 1 бит знака, 8 — экспоненты, 23 — мантиссы.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/KSS-oRLsPUnQ9Vypo7UZp.png)

Формула:
$$ v = (-1)^{\text{sign}} \cdot 2^{E-127} \cdot \left(1 + \sum_{i=1}^{23} b_{23-i}2^{-i}\right) $$

Ключевая идея вещественных типов: чем больше битов выделено под экспоненту, тем больший диапазон значений можно представить. Биты, оставшиеся для мантиссы, отвечают за точность, с которой представлены значения в диапазоне.

### Float16
Битовое представление: 1 бит знака, 5 — экспоненты и 10 — мантиссы.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/bX17lqakEY903HrSCZF-c.png)

Главная проблема float16 — маленький диапазон значений. Максимальное значение равно 65504, из-за чего тензоры активаций легко переполняются.

### Bfloat16, или brain float
Специальный формат данных, разработанный Google Brain. Можно рассматривать как аппроксимацию float32. Битовое представление такое: 1 бит знака, 8 — экспоненты и 7 — мантиссы.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/jeGGZP2DxQfXZZuB72iRD.png)

Обратите внимание, что число битов под экспоненту совпадает с представлением float32. Значит, bfloat16 представляет тот же диапазон значений, пусть и менее точно. Зато можно меньше опасаться переполнений в активациях.

Другая приятная особенность bf16 — возможность быстро конвертировать значения во float32. Магия работает благодаря сходному битовому представлению. К сожалению, пока что не всё железо работает с этим типом (особенно мобильное).

### TensorFloat32

Интересный 19-битный [тип данных](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) от NVidia. Поддерживается в архитектурах, начиная с NVidia Ampere (A-100). Битовое представление: 1 бит знака, 8 — экспоненты, 10 — мантиссы.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/ha7W9jLH-O1BvMrG5cAQf.png)

Ключевые особенности:
- число битов экспоненты совпадает с bfloat16, а значит и с float32;
- число битов мантиссы совпадает с float16.

Получился необычный, но точный и эффективный тип данных. Показывает отличные результаты по производительности вычислений и подходит для обучения моделей. Но существует только на современных видеокартах NVidia.

### E4M3 и E5M2
Новые 8-битные float. Предложены NVidia, ARM и Intel в статье [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433).
Авторы предлагают два возможных 8-битных вещественных значения:
- E4M3: 1 бит знака, 4 — экспоненты, 3 — мантиссы
- E5M2: 1 бит знака, 5 — экспоненты, 2 — мантиссы

Эксперименты показывают, что современные LLM и «картиночные» сети можно успешно инферить и даже обучать на таких типах данных. Ждём широкого распространения и поддержки в железе. Существуют и более радикальные идеи 4-битных вещественных значений: E2M1 и E3M0.

## [1. Where Did All the Memory Go?](https://arxiv.org/abs/1910.02054)

Let’s examine the memory consumption of the current training system. For example, a 1.5B parameter GPT-2 model requires 3GB (1.5B * 16bit) of memory for its weights (or parameters) in 16-bit precision, yet, it cannot be trained on a single GPU with 32GB memory using Tensorflow or PyTorch. One may wonder where all the memory goes. During model training, most of the memory is consumed by *model states*, i.e., tensors comprising of optimizer states, gradients, and parameters. Besides these model states, the rest of the memory is consumed by activations, temporary buffers and fragmented memory which we call *residual states*. We look at the memory consumption from both in details. 

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
Quantization is a procedure for compressing NN models by representing parameters and/or activations with a lower-bit representation such as 8-bit or 4-bit integer, instead of 32-bit or 16-bit floating point.

Cосредоточимся на линейной квантизации как на самом популярном и доказавшем свою эффективность методе.

### 2.1 Несимметричная и Симметричная линейная квантизация 
Взглянем сначала на иллюстрации:

**Несимметричная**:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/E1qaUh4uRmMXMfmxPlSiu.png)

**Симметричная**:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/QMMum7lBhmZlPj-BCANn8.png)

То есть, мы отображаем некоторый вещественный диапазон чисел в целочисленный. Сам процесс отображения можно проиллюстрировать так:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/JwT-g6-J31Hce_xylvAxD.png)

Где **S** и **Z** — это константы квантизации, то есть параметры, которые вычисляются в процессе. **S** - scale, отвечает за масштаб преобразования. **Z** - zero point, cответствует нулевому значению. 
- **Несимметричная**
  - \\(S = \frac {r_{max}-r_ {min}}{q_{max}-q_{min}} \\)
  - \\(Z = \left[q_{min} - \frac{r_{min}}{S}\right]\\)
  - \\(X_{quantized} = \left[\frac{X}{S} + Z\right]\\)
  - \\(X_{dequantized} = S(X_{quantized} - Z)\\)

- **Симметричная**
  - Границы квантизируемого диапазона определяют как максимальное по модулю квантизируемое значение.
  - \\(S = \frac{|r|_{max}}{2^{N-1} - 1} \\)
  - \\(Z = 0\\)
  - \\(X_{quantized} = \left[\frac{X}{S}\right]\\)
  - \\(X_{dequantized} = SX_{quantized}\\)
  - Чтобы тип получился симметричным, нужно отказаться от одного значения в квантизованном типе данных. Например, диапазон signed int8: [-128, 127] превратится в [-127, 127]

где \\([  ]\\) - округление.

Преимущества несимметричная квантизации — она умеет точнее и лучше справляться с асимметричными распределениями, в то время как симметричная квантизация выигрывает за счёт простоты и скорости. При таком подходе не нужно думать о хранении zero-point, а для деквантизации достаточно умножить тензор на константу.

Пример:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/6qS8qC8WTNdZFcnV16ufS.png)

Готово. На выходе мы получили 8-битный целочисленный тензор и константу квантизации 23,5. Теперь можно хранить меньший объём информации и при необходимости возвращаться к исходному 32-битному вещественному представлению с потерей точности.

### 2.2 Что квантизовать?

Стандартный подход — квантизовать веса модели. Никакие дополнительные манипуляции не нужны, просто воспользуйтесь формулами.

Также можно квантизовать выходы слоёв — активации. Для этого нужно оценить, какие значения встречаются в тензорах активаций. Как это сделать? Прогоняем через обученную нейросеть данные из обучающего датасета и собираем статистику. С помощью этой информации находим константы. Такой подход называют статической квантизацией.

А при динамической квантизации активации квантизуются на inference. Этот подход может дать лучшее качество, но с ним возможны трудности: в процессе inference искать константы придётся динамически. Это делает метод более сложным и вычислительно затратным, зато константы всегда остаются актуальными.

### 2.3 Когда квантизовать?

Готовить сеть к квантизации можно в процессе обучения, такой подход называется Quantize-Aware. Для этого в нейросеть встраивают специальные блоки и в ходе обучения эмулируют квантизованный inference.

Quantize-Aware-обучение сложное и требует больше вычислительных ресурсов, но на выходе получается модель, «приспособленная» к работе с квантизованными значениями и потенциально более точная.

В случае Post Training квантизуют уже обученную модель. Для квантизации активаций через обученную сеть дополнительно прогоняют данные из калибровочного датасета, собирают статистику по тензорам и потом квантизуют. Если квантизовать только веса, данные не нужны, так как вся информация уже есть в тензорах. Этот способ проще и быстрее, чем Quantize-Aware, но уступает ему в точности.

### 2.4 Гранулярность

Нейросеть можно квантизовать с разной гранулярностью. Самый плохой способ — квантизовать сразу всю сеть за раз. В этом случае у вас получится одна общая константа S на всю модель. Результат таких манипуляций, скорее всего, окажется неудовлетворительным.

Можно квантизовать тензоры по отдельности — тогда каждый тензор получит свои константы. А можно пойти дальше и в каждом тензоре квантизовать строки или столбцы. Соответственно, у каждой строки (столбца) в этом случае будет своя константа. Их придётся где-то хранить, зато вычисления будут точнее.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/xbY-NhjLaCY88RPi5PNee.png)

Также можно нарезать тензор на блоки небольшого размера — так получится ещё точнее. Этот подход позволяет бороться с выбросами в матрицах, о чём мы и поговорим дальше.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/S1SvT4tE5OEVvTskumu3c.png)

Итак, чем меньше гранулярность, тем меньше констант нужно хранить, и наоборот — чем выше гранулярность, тем ближе результаты квантизованных вычислений к исходным.

### 2.5 Типы данных

В квантизованных нейросетевых моделях обычно присутствуют два типа данных:

- **Quantized type** — в этом типе хранят тензоры
- **Computation type** — в этом типе проводят вычисления.

К сожалению, эти два типа не всегда совпадают. Например, ваше железо может не поддерживать операции в хитром quantized type. Эффективных кернелов перемножения матриц под квантизованный тип может просто не существовать. В таких случаях перед вычислениями матрицу нужно конвертировать в computation type. Также computation type позволяет избежать проблем с переполнением в активациях, так как перемножение 8-битных чисел наверняка приведёт к выходу за границы типа.

### 2.6 Проблема выбросов
Посмотрим на пример симметричной квантизации

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/AHIUqmayD-GX7PpEPirpg.png)

Что получится, если во входной тензор попадёт выброс?

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/M-NJYW1SuSDNU4qiHa5Dv.png)

Веса «склеились» в узкий диапазон и стали неотличимы. Качество модели потеряно. Так единственный выброс испортил всю матрицу.

Когда число параметров становится больше и больше, стандартные техники квантизации перестают работать. При переходе границы в 6,7 миллиардов параметров квантизованные модели [теряют всё качество](https://arxiv.org/abs/2208.07339). Происходит это из-за растущего числа выбросов в матрицах

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/juzSha_nIdR3znammiwgO.png)









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

Each parameter now requires 16 bytes, so training Llama 70B would still require approximately 1040 GB. You might wonder why the memory usage remains the same as in float32. The reason is that while we use 2 bytes (16 bits) for weights and gradients in float16, we also store a copy of the weights in float32, adding 4 bytes per parameter. But. there might be the major saving come from reduced activation memory.

Also the significant advantage of Mixed Precision lies in computation speed—most calculations are now done in float16, which considerably speeds up the training process.

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

Now, let’s understand the entire QLoRA process (L1 and L2 in the formulas correspond to B and A in the figure):
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

#### 1.5. Flash Attention

Scaling the transformer architecture is heavily bottlenecked by the self-attention mechanism, which has quadratic time and memory complexity. Recent developments in accelerator hardware mainly focus on enhancing compute capacities and not memory and transferring data between hardware. This results in attention operation having a memory bottleneck.

Standard attention mechanism uses High Bandwidth Memory (HBM) to store, read and write keys, queries and values. HBM is large in memory, but slow in processing, meanwhile SRAM is smaller in memory, but faster in operations. In the standard attention implementation, the cost of loading and writing keys, queries, and values from HBM is high. It loads keys, queries, and values from HBM to GPU on-chip SRAM, performs a single step of the attention mechanism, writes it back to HBM, and repeats this for every single attention step.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/g4EnH54JxS_ZFvNgtVNce.png)

[**FlashAttention**](https://tridao.me/publications/flash3/flash3.pdf) is an algorithm that reorders the attention computation and leverages tiling and recomputation to significantly speed it up and reduce memory usage from quadratic to linear in sequence length. It uses tiling to load blocks of inputs from HBM (GPU memory) to SRAM (fast cache), perform attention with respect to that block, and update the output in HBM. By not writing the large intermediate attention matrices to HBM, we reduce the amount of memory reads/writes, which brings 2-4x wallclock time speedup.

Diagram of FlashAttention forward pass: with tiling and softmax rescaling, we operate by blocks and avoid having to read/write from HBM, while obtaining the correct output with no approximation.
![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/_Xf7ZPpoX6o-17ARq6B4E.png)

For FP16:

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/-VbVssfm8mkToIrFxJgWk.png)

We have come a long way. Of the hard stuff, only one topic remains to be dealt with - distributed computing. Before them, let's briefly discuss a few more small methods that can slightly improve your performance

#### 1.6. Gradient Accumulation

**Gradient accumulation** is a technique where you can train on bigger batch sizes than your machine would normally be able to fit into memory. This is done by accumulating gradients over several batches, and only stepping the optimizer after a certain number of batches have been performed.

For instance, if the gradient accumulation factor is set to 2, the process works as follows: We first calculate the gradient on one batch, which gives us a direction on the [loss function landscape](https://losslandscape.com/). Instead of updating the model weights immediately, we calculate another gradient from the next batch, obtaining a potentially different direction. By adding these two gradients together, we find a more accurate path in the loss landscape. To ensure the final update step is properly scaled, we divide the accumulated gradient by the number of batches, preventing any artificial inflation of the step size.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/660710b03ef451aa2bab8971/UJ2nzxFp7EUquy61gDnBZ.png)

This technique is particularly useful when only small batch sizes can fit into memory, which might otherwise lead to overly noisy updates and less stable training.

#### 1.7. 8-bit optimizers

Помните, как много оказывается потребляет памяти оптимизатор? Давайте чуть глубже поймем, почему. Сначала вспомним формулу простейшего SGD оптимизатора (x - это веса):
$$ x_{k+1} = x_k - \alpha \nabla f(x_k) $$

Как видим, здесь нам нужны только градиенты по весам. Но такой оптимизатор 

$$ v_{k+1} = \beta_1 v_k + (1 - \beta_1) \nabla f(x_k) $$
$$ G_{k+1} = \beta_2 G_k + (1 - \beta_2) (\nabla f(x_k))^2 $$
$$ x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_{k+1} + \varepsilon}} v_{k+1} $$
