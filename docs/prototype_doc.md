# Working Prototype Known Problems Report

Product name: Text Augmentation Using Pre-Trained Transformers

Team name: TFRL

Date: 7/21/20

## List of functions not working correctly

1.  The `trl` package can only work with `transformers==2.6.0` because the author changed the package dependency during our project.
2.  The parameters of the  `prepare_data` function are kind of not user friendly.
3.  The user has to install `nlp` from source because the `.from_pandas()` function is not available in the latest PyPI version.

