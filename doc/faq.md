# FAQ

## What input MRI is required?
Requirements are similar to FreeSurfer: High quality 3D T1-weighted MRI with 1mm isotropic resolution acquired at 3T or 1.5T. Other resolutions are automatically re-sampled to 1mm isovoxels (inspect the results in this case). Recommended sequences are those with an excellent gray/white matter contrast like Siemens [MP-RAGE](https://doi.org/10.1002/mrm.1910150117) (ideally with TI=1100ms) as proposed by [ADNI](https://doi.org/10.1002/jmri.21049), or [MDEFT](https://doi.org/10.1016/j.neuroimage.2003.09.062). Similar sequences should work as well, just try it out.

DL+DiReCT expects the input without skull (a.k.a. brain extracted / skull-stripped). You may use the ```--bet``` options to skull-strip the input using [HD-BET](https://github.com/NeuroAI-HD/HD-BET/) before processing. If the input is already skull-stripped, ensure background voxels are zero.


## Can I use an MP2RAGE sequence as input?
In principal yes, DL+DiReCT should work fine on a (brain extracted) MP2RAGE sequence. However, the characteristic "salt & pepper" noise that is a result of numerical instability that amplifies noise in areas with a very low SNR like background may require special treatment for brain extraction. For brain extraction using HD-BET, you may use the the 2<sup>nd</sup> inversion recovery (proton density-weighted) image from an MP2RAGE sequence to generate the brain mask which is then applied to the original unified image. This is available in ```dl+direct``` with the options
```bash
--bet --mp2rage-inv2 <path_to_inv2.nii.gz>
```

__Note__: A de-noised image can also be calculated by multiplying the unified image with the 2<sup>nd</sup> inversion recovery ([Fujimoto et al., 2014](https://doi.org/10.1016/j.neuroimage.2013.12.012)). This alternative (```bet.py --mp2rage-inv2x```) might work for other brain extraction algorithms but seems not optimal for HD-BET. Visual inspection of the brain mask is recommended. DL+DiReCT should be run on the original (unified) image with the brain mask applied and not on the de-noised version.
