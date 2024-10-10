# GPT2 From Scratch

## Цель проекта
1. Собрать с нуля из спичек и желудей архитектуру GPT2, чтобы лучше понять подкапотную LLM
2. Сделать pretrain собранной модели на небольшом (10B токенов) дампе датасета FineWeb_edu, используя различные методы оптимизации, чтобы ускорить этот длительный процесс.

# Результаты
Все цели выполнены
1. Если хорошо знать теорию трансформеров, то имплементация архитектуры не представляет большого труда. Самое сложное это сделать эффективное распараллеливание в блоке multi head attention и соблюсти все тонкости оригинальной инициализации.
2. Обучать LLM почти на сыром пайторче - дело на из простых. Для распределенных вычислений я пользуюсь библиотекой accelerate от HF. Для проверки результатов я сверял свои метрики с метриками Андрея Карпатого, всё сошлось.



# Requirements

- OS: Ubuntu 24.05 CUDA 12.4
- Miniconda: Conda 24.7.1 Python 3.12.4 released Aug 22, 2024. https://docs.anaconda.com/miniconda/#quick-command-line-install
- conda create -n myenv python=3.11
- PyTorch 2.4.1: conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia    https://pytorch.org/get-started/locally/
- numpy==2.1.2 transformers==4.45.1 datasets==3.0.1 accelerate==1.0.0