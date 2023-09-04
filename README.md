# Applied Optimization Methods for Inverse Problems

Repository to solve the Helsinki Tomography Challenge 2022, while also exploring a wide range of optimization methods
for inverse problems.

This was mainly done as part of the practical course "Applied Optimization Method for Inverse Problems" offered during
the summer term 2023 at TUM.

## Getting started

#### Poetry

The easiest and recommended way to install is using `poetry` (see
[here](https://python-poetry.org/)). Once you've installed `poetry`,
run `poetry install` from the root directory.

Next, we miss the dependency on `elsa` (see
[here](https://gitlab.lrz.de/IP/elsa)), the tool for tomographic
reconstruction. First run `poetry shell`, which will activate the virtual
environment created by `poetry`, then clone `elsa` to a new directory,
move into the directory and run `pip install . --verbose` (the `--verbose` is
optional, but then you'll see the progress).

From now, you can either activate the virtual environment by running `poetry shell`,
or you run `poetry run python myscript`, which will activate the environment for
that single command.

#### Classic

If you do not want to use `poetry`, you can use virtual environments.
From the root directory of this repository run `python -m venv /path/to/venv`, activate
it using `source /path/to/venv/bin/activate`, and then install everything with
`pip install --editable .` (from the root directory). Feel free to use the requirement.txt file.

Then again you need to install `elsa`. Follow the steps described above in
the `poetry` section.

### Getting the data for the Helsinki Tomography Challenge

To get the dataset for the challenge, head over to
[Zenodo](https://doi.org/10.5281/zenodo.7418878) and download the `htc2022_test_data.zip`. Extract it to a folder and
you should be good to go.
In the folder you will see a couple of different files.

`.mat` files contain the actual measurements/sinogram, which are needed for
reconstruction. There are the full measurements, one with limited number of
projections and example reconstructions of both. Further, there are segmented
files, which show the required binary thresholding segmentation done to
evaluate the score (again for full data and limited). However, for convenience the scoring is already implemented in
challenge/utils.py and can be used that way.
Finally, there are a couple of example images.

### Troubleshooting

If you have trouble installing `elsa` , see the README
of `elsa`. If you use an Ubuntu based distro and want to use CUDA, you might
need to set `CUDA_HOME`, to wherever CUDA is installed.

Please note, that you do not need CUDA, but it might speed up your
reconstructions quite dramatically.

## Results

### Process

For the challenge, I decided to focus my efforts initially on the phantom A with difficulty 7, look at the
various angle cases and then try to apply the gained knowledge to the other phantoms. My procedure was basically
the following:
  1. Determine best starting point with FBP (Filtered BackProjection)
  2. Choose promising formulations (Usually a linear system with some regularization)
  3. Determine formulation/algorithm combinations and optimizer setup
  4. Compare formulations and explore formulation parameters
  5. Trim combinations and formulations
  6. Extrapolate to other phantoms and Squeeze

### Results

My final best scores and their associated parameters are the following (Scores go from 0 to 1):

ARC: 360

 Phantom | Difficulty | Score      | i   | n_iter | Formulation      | Method             
---------|------------|------------|-----|--------|------------------|--------------------
 a       | 1          | 0.99911337 | 111 | 452    | LSDfRFormulation | optimized_gradient 
 a       | 2          | 0.99836131 | 172 | 303    | LSDfRFormulation | optimized_gradient 
 a       | 3          | 0.99871893 | 108 | 320    | LSDfRFormulation | optimized_gradient 
 a       | 4          | 0.99780540 | 200 | 308    | LSDfRFormulation | optimized_gradient 
 a       | 5          | 0.99710220 | 118 | 305    | LSDfRFormulation | optimized_gradient 
 a       | 6          | 0.99777936 | 131 | 309    | LSDfRFormulation | optimized_gradient 
 a       | 7          | 0.99592590 | 133 | 291    | LSDfRFormulation | optimized_gradient 
 b       | 1          | 0.99889755 | 110 | 180    | LSDfRFormulation | optimized_gradient 
 b       | 2          | 0.99765429 | 213 | 314    | LSDfRFormulation | optimized_gradient 
 b       | 3          | 0.99780123 | 156 | 303    | LSDfRFormulation | optimized_gradient 
 b       | 4          | 0.99709347 | 179 | 325    | LSDfRFormulation | optimized_gradient 
 b       | 5          | 0.99710042 | 172 | 295    | LSDfRFormulation | optimized_gradient 
 b       | 6          | 0.99804756 | 92  | 309    | LSDfRFormulation | optimized_gradient 
 b       | 7          | 0.99701880 | 92  | 297    | LSDfRFormulation | optimized_gradient 
 c       | 1          | 0.99832031 | 152 | 321    | LSDfRFormulation | optimized_gradient 
 c       | 2          | 0.99895954 | 100 | 307    | LSDfRFormulation | optimized_gradient 
 c       | 3          | 0.99808751 | 93  | 321    | LSDfRFormulation | optimized_gradient 
 c       | 4          | 0.99817051 | 95  | 326    | LSDfRFormulation | optimized_gradient 
 c       | 5          | 0.99698048 | 126 | 310    | LSDfRFormulation | optimized_gradient 
 c       | 6          | 0.99749203 | 151 | 183    | LSDfRFormulation | optimized_gradient 
 c       | 7          | 0.99792990 | 160 | 312    | LSDfRFormulation | optimized_gradient 

ARC: 90

 Phantom | Difficulty | Score      | i   | n_iter | Formulation      | Method                      
---------|------------|------------|-----|--------|------------------|-----------------------------
 a       | 1          | 0.98315684 | 0   | 48     | LassoFormulation | optimized_proximal_gradient 
 a       | 2          | 0.97160876 | 110 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 3          | 0.94341835 | 412 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 4          | 0.95396707 | 124 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 5          | 0.94761390 | 467 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 6          | 0.88822590 | 490 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 7          | 0.88385510 | 421 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 1          | 0.95903665 | 130 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 2          | 0.95546577 | 125 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 3          | 0.92124738 | 0   | 50     | LassoFormulation | optimized_proximal_gradient 
 b       | 4          | 0.85531617 | 105 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 5          | 0.93823663 | 434 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 6          | 0.83720253 | 528 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 7          | 0.87818244 | 142 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 1          | 0.95233803 | 108 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 2          | 0.95991236 | 0   | 49     | LassoFormulation | optimized_proximal_gradient 
 c       | 3          | 0.91283543 | 454 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 4          | 0.93823993 | 108 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 5          | 0.87203141 | 483 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 6          | 0.93357335 | 120 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 7          | 0.94400170 | 107 | 1000   | LSDfRFormulation | optimized_gradient          

ARC: 60

 Phantom | Difficulty | Score      | i   | n_iter | Formulation      | Method                      
---------|------------|------------|-----|--------|------------------|-----------------------------
 a       | 1          | 0.97581547 | 0   | 43     | LassoFormulation | optimized_proximal_gradient 
 a       | 2          | 0.94439732 | 428 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 3          | 0.86466503 | 495 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 4          | 0.81652094 | 617 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 5          | 0.86991124 | 523 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 6          | 0.86431890 | 437 | 1000   | LSDfRFormulation | optimized_gradient          
 a       | 7          | 0.84978890 | 506 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 1          | 0.92504951 | 507 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 2          | 0.89841847 | 534 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 3          | 0.78610247 | 491 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 4          | 0.78565684 | 428 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 5          | 0.86656066 | 523 | 1000   | LSDfRFormulation | optimized_gradient          
 b       | 6          | 0.79117909 | 0   | 43     | LassoFormulation | optimized_proximal_gradient 
 b       | 7          | 0.75971128 | 540 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 1          | 0.89991305 | 538 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 2          | 0.86561725 | 609 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 3          | 0.86056884 | 432 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 4          | 0.78753502 | 488 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 5          | 0.83520899 | 0   | 48     | LassoFormulation | optimized_proximal_gradient 
 c       | 6          | 0.86982834 | 451 | 1000   | LSDfRFormulation | optimized_gradient          
 c       | 7          | 0.87492935 | 512 | 1000   | LSDfRFormulation | optimized_gradient          

ARC: 30

 Phantom | Difficulty | Score      | i | n_iter | Formulation                 | Method 
---------|------------|------------|---|--------|-----------------------------|--------
 a       | 1          | 0.97522754 | 2 | 66     | TVRegularizationFormulation | admm   
 a       | 2          | 0.86904071 | 3 | 66     | TVRegularizationFormulation | admm   
 a       | 3          | 0.82638661 | 4 | 573    | ElasticNetFormulation       | admm   
 a       | 4          | 0.78936320 | 4 | 66     | TVRegularizationFormulation | admm   
 a       | 5          | 0.85047692 | 3 | 539    | ElasticNetFormulation       | admm   
 a       | 6          | 0.81374212 | 3 | 67     | TVRegularizationFormulation | admm   
 a       | 7          | 0.80093117 | 4 | 66     | TVRegularizationFormulation | admm   
 b       | 1          | 0.90751182 | 6 | 66     | TVRegularizationFormulation | admm   
 b       | 2          | 0.86670303 | 4 | 540    | ElasticNetFormulation       | admm   
 b       | 3          | 0.74334419 | 4 | 574    | ElasticNetFormulation       | admm   
 b       | 4          | 0.65569374 | 4 | 67     | TVRegularizationFormulation | admm   
 b       | 5          | 0.84435297 | 3 | 66     | TVRegularizationFormulation | admm   
 b       | 6          | 0.78933489 | 3 | 67     | TVRegularizationFormulation | admm   
 b       | 7          | 0.74354104 | 4 | 66     | TVRegularizationFormulation | admm   
 c       | 1          | 0.80041927 | 4 | 66     | TVRegularizationFormulation | admm   
 c       | 2          | 0.84188645 | 3 | 142    | ElasticNetFormulation       | admm   
 c       | 3          | 0.81581281 | 4 | 66     | TVRegularizationFormulation | admm   
 c       | 4          | 0.67804736 | 3 | 66     | TVRegularizationFormulation | admm   
 c       | 5          | 0.82568818 | 4 | 66     | TVRegularizationFormulation | admm   
 c       | 6          | 0.77035164 | 4 | 67     | TVRegularizationFormulation | admm   
 c       | 7          | 0.76017163 | 4 | 66     | TVRegularizationFormulation | admm   

And finally, the associated reconstructions compared to ground truth:

![final_reconstructions.png](img%2Ffinal_reconstructions.png)
