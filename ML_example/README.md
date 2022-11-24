# ML experiment

Original code and data taken from https://github.com/yaringal/DropoutUncertaintyExps. 

Adaption: Removed dropout layers and included IDR fitting and smoothing. 

Code requires the function *smooth_crps* which is available here: https://github.com/evwalz/crpsmixture

To reproduce experiment run:

``` python experiment.py --dir energy -e 1 -nh 1 ```
