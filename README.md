# Group Distribution Robust Optimization project in Generative models

This project takes initiatives of modifying the gradient direction based on loss per group in order to make models more robust to imbalanced datasets.

The original implementation of Group loss is taken from the Github repo https://github.com/kohpangwei/group_DRO

For more details, please refer to the paper "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization" 

**Paper**: https://arxiv.org/abs/1911.08731

**Authors**: Shiori Sagawa*, Pang Wei Koh*, Tatsunori Hashimoto, and Percy Liang

This project uses the framework MMagic by OpenMMlab.

## Super resolution

Problem with the state-of-the-art Super Resolution (SR) trained on dataset with a common method of interpolation to resize images **Bicubic**. In our test with other interpolation methods, e.g. k-nearest neighbor, bilinear, the performance of these modes deteriorates quite quickly.



