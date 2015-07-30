[![Build Status](https://travis-ci.org/hassaku/deformable-supervised-nmf.png)](https://travis-ci.org/hassaku/deformable-supervised-nmf)

Deformable Supervised NMF
=======

# Usage

```
dsnmf = DeformableSupervisedNMF(supervised_components_list=[5, 5], unknown_componets=5,
                                supervised_max_iter_list=[1000]*2, unknown_max_iter=5000,
                                eta=0.01, mu_list=[0.01, 0.01, 0.01, 0.01],
                                X_list=[pretraining_data1, pretraining_data2])

deformed_features1, deformed_features2, unknown_features = dsnmf.fit_transform(composing_data)
```

![sample result](https://raw.github.com/wiki/hassaku/deformable-supervised-nmf/screen_shots/result.png)
