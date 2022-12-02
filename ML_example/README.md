# ML experiment

Original code and data taken from https://github.com/yaringal/DropoutUncertaintyExps. To compute CRPS for MC dropout use function *crps_mixnorm_mc* from https://github.com/evwalz/crpsmixture 

Adaptation of original code for EasyUQ: Removed dropout layers and included IDR fitting and smoothing. 

To reproduce experiment run:

``` python experiment.py --dir energy -e 1 -nh 1 ```

Results saved in [UCI_Dataset](https://github.com/evwalz/easyuq/tree/main/ML_example/UCI_Datasets)
