A Learned Proximal Alternating Minimization Algorithm and Its Induced Network for a Class of Two-Block Nonconvex and Nonsmooth Optimization

https://link.springer.com/article/10.1007/s10915-025-02874-5

https://www.researchgate.net/publication/385721528_A_Learned_Proximal_Alternating_Minimization_Algorithm_and_Its_Induced_Network_for_a_Class_of_Two-block_Nonconvex_and_Nonsmooth_Optimization

https://arxiv.org/abs/2411.06333

The data.py is not uploaded. You can create a dataloader yourself. 

In this code, k1 is kspace image for T1; k2 is kspace image for T2. t1 is ground truth image for T1; k2 is ground truth image for T2. kx1 is undersampled image for T1 and kx2 is undersampled image for T2. kx1 and kx2 are updated each epoch.

Use "python main.py" to train and test. When testing, command the training part in main.py.

This code is only used for batch_size=1.
