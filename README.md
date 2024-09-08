# MIFNO
Train a Multiple-Input Fourier Neural Operator (MIFNO) to predict the solution of 3D source-dependent Partial Differential Equations (PDEs). The MIFNO is described in the article [Multiple-Input Fourier Neural Operator (MIFNO) for source-dependent 3D elastodynamics](https://arxiv.org/abs/2404.10115). It extends the 3D Factorized Fourier Neural Operator (F-FNO, [Tran et al., 2023](https://openreview.net/forum?id=tmIiMPl4IPa)) to PDEs with a source term. As such, the MIFNO contains a dedicated *source branch* that takes as input a vector of source parameters. 

![MIFNO](https://github.com/user-attachments/assets/e08b86b4-2374-41ee-8e4f-7a4ab8019805)

## Data
The MIFNO is trained on the [HEMEW<sup>S</sup>-3D database](https://doi.org/10.57745/LAI6YU) that contains 30,000 simulations of the 3D elastic wave equation in heterogeneous media with different sources. The folder `data` contains the codes to pre-process the data and save them to a format convenient for machine learning applications.

## Training
The folder `models` contains the models' architectures for the MIFNO and the F-FNO. The main code is `train.py` that serves to train the models. Scripts `launch_ffno.sh` and `launch_mifno.sh` show how to define the variables to train the models. 

## Evaluation
The folder `models` contains the code `evalute_metrics.py` to evaluate the predictions with different metrics: relative Mean Absolute Error (rMAE), relative Root Mean Square Error (rRMSE) and frequency biases.
The difference between two seismic signals is commonly assessed with the Envelope and Phase Goodness-Of-Fit (GOF) measures ([Kristekova et al., 2009](https://doi.org/10.1111/j.1365-246X.2009.04177.x)). The code `evaluate_gof.py` computes the envelope and phase GOFs of MIFNO prediction. It requires the [obspy](https://docs.obspy.org/) package and runs in parallel.

## Visualization
A jupyter notebook `Plots.ipynb` illustrates the input and output data. 


## References
If you use this code, please cite 
```
@misc{lehmannMultipleInputFourierNeural2024,
  title = {Multiple-{{Input Fourier Neural Operator}} ({{MIFNO}}) for Source-Dependent {{3D}} Elastodynamics},
  author = {Lehmann, Fanny and Gatti, Filippo and Clouteau, Didier},
  year = {2024},
  number = {arxiv:2404.10115},
  eprint = {2404.10115},
  publisher = {arXiv},
  url = {https://arxiv.org/abs/2404.10115},
  archiveprefix = {arXiv},
  annotation = {10.48550/ARXIV.2404.10115},
}
```

If you use the HEMEW<sup>S</sup>-3D database, please cite
```
@misc{lehmannPhysicsbasedSimulations3D2023a,
  title = {Physics-Based {{Simulations}} of {{3D Wave Propagation}} with {{Source Variability}}: \${{HEMEW}}{\textasciicircum}{{S-3D}}\$},
  shorttitle = {Physics-Based {{Simulations}} of {{3D Wave Propagation}} with {{Source Variability}}},
  author = {Lehmann, Fanny},
  year = {2023},
  publisher = {[object Object]},
  doi = {10.57745/LAI6YU},
  url = {https://doi.org/10.57745/LAI6YU}
}
```
and 
```
@article{lehmannSyntheticGroundMotions2024,
  title = {Synthetic Ground Motions in Heterogeneous Geologies from Various Sources: The {{HEMEW}}{\textbackslash}textsuperscript\{\vphantom\}{{S}}\vphantom\{\}-{{3D}} Database},
  author = {Lehmann, F. and Gatti, F. and Bertin, M. and Clouteau, D.},
  year = {2024},
  journal = {Earth System Science Data},
  volume = {16},
  number = {9},
  pages = {3949--3972},
  doi = {10.5194/essd-16-3949-2024},
  url = {https://essd.copernicus.org/articles/16/3949/2024/}
}
```
