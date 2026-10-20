# A. Information diagnostics — every representation and window

HuGaDB: 10 observed classes, 18 subjects, wearable IMU channels. LARa: 8 classes including None, 16 subjects, optical motion-capture channels. Cross-dataset comparisons differ in class count, subject count and modality; HuGaDB cannot establish body-shape effects.

Observed values are mean ± population SD over the three clustering seeds. Reference mean is the mean of the three 20-permutation means. Reference SD column is the mean of the three within-cell permutation SDs; it is not a seed SD or a pooled 60-permutation SD. Entropies and MI are in nats.

## hugadb, latent, W=15

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.9396 ± 0.0010 | 0.0005 | 0.0001 |
| 10 | mi_unit_subject | 0.1754 ± 0.0010 | 0.0010 | 0.0001 |
| 10 | cmi_unit_subject_given_action | 0.2086 ± 0.0007 | 0.0046 | 0.0002 |
| 10 | h_unit | 1.9657 ± 0.0009 | 1.9657 | 0.0000 |
| 10 | h_action_given_unit | 1.0880 ± 0.0010 | 2.0271 | 0.0001 |
| 10 | h_subject_given_unit | 2.6688 ± 0.0010 | 2.8432 | 0.0001 |
| 10 | nmi_unit_action | 0.4706 ± 0.0004 | 0.0003 | 0.0000 |
| 10 | nmi_unit_subject | 0.0729 ± 0.0004 | 0.0004 | 0.0000 |
| 10 | ami_unit_action | 0.4705 ± 0.0004 | 0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0725 ± 0.0004 | -0.0000 | 0.0000 |
| 20 | mi_unit_action | 1.0137 ± 0.0145 | 0.0011 | 0.0001 |
| 20 | mi_unit_subject | 0.2236 ± 0.0058 | 0.0021 | 0.0001 |
| 20 | cmi_unit_subject_given_action | 0.2538 ± 0.0059 | 0.0087 | 0.0003 |
| 20 | h_unit | 2.4164 ± 0.0226 | 2.4164 | 0.0000 |
| 20 | h_action_given_unit | 1.0140 ± 0.0145 | 2.0265 | 0.0001 |
| 20 | h_subject_given_unit | 2.6207 ± 0.0058 | 2.8421 | 0.0001 |
| 20 | nmi_unit_action | 0.4562 ± 0.0047 | 0.0005 | 0.0001 |
| 20 | nmi_unit_subject | 0.0850 ± 0.0019 | 0.0008 | 0.0001 |
| 20 | ami_unit_action | 0.4559 ± 0.0047 | 0.0000 | 0.0001 |
| 20 | ami_unit_subject | 0.0842 ± 0.0019 | -0.0000 | 0.0001 |
| 50 | mi_unit_action | 1.1373 ± 0.0054 | 0.0029 | 0.0002 |
| 50 | mi_unit_subject | 0.5055 ± 0.0798 | 0.0055 | 0.0003 |
| 50 | cmi_unit_subject_given_action | 0.5409 ± 0.0827 | 0.0200 | 0.0005 |
| 50 | h_unit | 3.1988 ± 0.0905 | 3.1988 | 0.0000 |
| 50 | h_action_given_unit | 0.8903 ± 0.0054 | 2.0248 | 0.0002 |
| 50 | h_subject_given_unit | 2.3387 ± 0.0798 | 2.8388 | 0.0003 |
| 50 | nmi_unit_action | 0.4354 ± 0.0091 | 0.0011 | 0.0001 |
| 50 | nmi_unit_subject | 0.1670 ± 0.0237 | 0.0018 | 0.0001 |
| 50 | ami_unit_action | 0.4347 ± 0.0091 | -0.0000 | 0.0001 |
| 50 | ami_unit_subject | 0.1655 ± 0.0237 | 0.0000 | 0.0001 |
| 100 | mi_unit_action | 1.2405 ± 0.0095 | 0.0059 | 0.0003 |
| 100 | mi_unit_subject | 0.8771 ± 0.0348 | 0.0111 | 0.0004 |
| 100 | cmi_unit_subject_given_action | 0.9207 ± 0.0359 | 0.0358 | 0.0006 |
| 100 | h_unit | 3.9332 ± 0.0500 | 3.9332 | 0.0000 |
| 100 | h_action_given_unit | 0.7871 ± 0.0095 | 2.0218 | 0.0003 |
| 100 | h_subject_given_unit | 1.9672 ± 0.0348 | 2.8331 | 0.0004 |
| 100 | nmi_unit_action | 0.4163 ± 0.0057 | 0.0020 | 0.0001 |
| 100 | nmi_unit_subject | 0.2588 ± 0.0085 | 0.0033 | 0.0001 |
| 100 | ami_unit_action | 0.4151 ± 0.0057 | -0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.2563 ± 0.0086 | -0.0000 | 0.0001 |
| 500 | mi_unit_action | 1.4278 ± 0.0030 | 0.0309 | 0.0006 |
| 500 | mi_unit_subject | 1.6803 ± 0.0147 | 0.0589 | 0.0009 |
| 500 | cmi_unit_subject_given_action | 1.7154 ± 0.0163 | 0.1174 | 0.0012 |
| 500 | h_unit | 5.3784 ± 0.0170 | 5.3784 | 0.0000 |
| 500 | h_action_given_unit | 0.5998 ± 0.0030 | 1.9967 | 0.0006 |
| 500 | h_subject_given_unit | 1.1640 ± 0.0147 | 2.7853 | 0.0009 |
| 500 | nmi_unit_action | 0.3856 ± 0.0009 | 0.0083 | 0.0002 |
| 500 | nmi_unit_subject | 0.4087 ± 0.0028 | 0.0143 | 0.0002 |
| 500 | ami_unit_action | 0.3804 ± 0.0009 | 0.0000 | 0.0002 |
| 500 | ami_unit_subject | 0.4001 ± 0.0029 | -0.0000 | 0.0002 |
| 1000 | mi_unit_action | 1.4799 ± 0.0028 | 0.0609 | 0.0008 |
| 1000 | mi_unit_subject | 1.9138 ± 0.0049 | 0.1172 | 0.0011 |
| 1000 | cmi_unit_subject_given_action | 1.9369 ± 0.0043 | 0.1873 | 0.0015 |
| 1000 | h_unit | 5.8526 ± 0.0076 | 5.8526 | 0.0000 |
| 1000 | h_action_given_unit | 0.5478 ± 0.0028 | 1.9667 | 0.0008 |
| 1000 | h_subject_given_unit | 0.9304 ± 0.0049 | 2.7271 | 0.0011 |
| 1000 | nmi_unit_action | 0.3756 ± 0.0011 | 0.0155 | 0.0002 |
| 1000 | nmi_unit_subject | 0.4401 ± 0.0009 | 0.0269 | 0.0003 |
| 1000 | ami_unit_action | 0.3657 ± 0.0011 | -0.0001 | 0.0002 |
| 1000 | ami_unit_subject | 0.4246 ± 0.0009 | -0.0000 | 0.0003 |

## hugadb, latent, W=30

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.9785 ± 0.0047 | 0.0011 | 0.0002 |
| 10 | mi_unit_subject | 0.2054 ± 0.0014 | 0.0019 | 0.0002 |
| 10 | cmi_unit_subject_given_action | 0.2264 ± 0.0010 | 0.0090 | 0.0004 |
| 10 | h_unit | 1.9407 ± 0.0182 | 1.9407 | 0.0000 |
| 10 | h_action_given_unit | 1.0474 ± 0.0047 | 2.0249 | 0.0002 |
| 10 | h_subject_given_unit | 2.6389 ± 0.0014 | 2.8424 | 0.0002 |
| 10 | nmi_unit_action | 0.4934 ± 0.0046 | 0.0005 | 0.0001 |
| 10 | nmi_unit_subject | 0.0859 ± 0.0009 | 0.0008 | 0.0001 |
| 10 | ami_unit_action | 0.4931 ± 0.0046 | 0.0000 | 0.0001 |
| 10 | ami_unit_subject | 0.0851 ± 0.0009 | -0.0000 | 0.0001 |
| 20 | mi_unit_action | 0.9978 ± 0.0050 | 0.0023 | 0.0002 |
| 20 | mi_unit_subject | 0.2282 ± 0.0064 | 0.0043 | 0.0003 |
| 20 | cmi_unit_subject_given_action | 0.2627 ± 0.0071 | 0.0158 | 0.0005 |
| 20 | h_unit | 2.3805 ± 0.0104 | 2.3805 | 0.0000 |
| 20 | h_action_given_unit | 1.0282 ± 0.0050 | 2.0237 | 0.0002 |
| 20 | h_subject_given_unit | 2.6161 ± 0.0064 | 2.8400 | 0.0003 |
| 20 | nmi_unit_action | 0.4529 ± 0.0031 | 0.0010 | 0.0001 |
| 20 | nmi_unit_subject | 0.0874 ± 0.0026 | 0.0016 | 0.0001 |
| 20 | ami_unit_action | 0.4523 ± 0.0031 | -0.0000 | 0.0001 |
| 20 | ami_unit_subject | 0.0859 ± 0.0026 | 0.0000 | 0.0001 |
| 50 | mi_unit_action | 1.1312 ± 0.0064 | 0.0059 | 0.0004 |
| 50 | mi_unit_subject | 0.6226 ± 0.0154 | 0.0109 | 0.0005 |
| 50 | cmi_unit_subject_given_action | 0.6641 ± 0.0165 | 0.0366 | 0.0010 |
| 50 | h_unit | 3.3156 ± 0.0032 | 3.3156 | 0.0000 |
| 50 | h_action_given_unit | 0.8947 ± 0.0064 | 2.0200 | 0.0004 |
| 50 | h_subject_given_unit | 2.2218 ± 0.0154 | 2.8334 | 0.0005 |
| 50 | nmi_unit_action | 0.4235 ± 0.0022 | 0.0022 | 0.0001 |
| 50 | nmi_unit_subject | 0.2021 ± 0.0051 | 0.0036 | 0.0002 |
| 50 | ami_unit_action | 0.4223 ± 0.0022 | 0.0000 | 0.0001 |
| 50 | ami_unit_subject | 0.1993 ± 0.0051 | -0.0001 | 0.0002 |
| 100 | mi_unit_action | 1.2675 ± 0.0022 | 0.0121 | 0.0006 |
| 100 | mi_unit_subject | 0.8960 ± 0.0328 | 0.0226 | 0.0006 |
| 100 | cmi_unit_subject_given_action | 0.9325 ± 0.0303 | 0.0617 | 0.0009 |
| 100 | h_unit | 3.9001 ± 0.0321 | 3.9001 | 0.0000 |
| 100 | h_action_given_unit | 0.7584 ± 0.0022 | 2.0138 | 0.0006 |
| 100 | h_subject_given_unit | 1.9483 ± 0.0328 | 2.8217 | 0.0006 |
| 100 | nmi_unit_action | 0.4278 ± 0.0016 | 0.0041 | 0.0002 |
| 100 | nmi_unit_subject | 0.2657 ± 0.0086 | 0.0067 | 0.0002 |
| 100 | ami_unit_action | 0.4254 ± 0.0016 | 0.0000 | 0.0002 |
| 100 | ami_unit_subject | 0.2607 ± 0.0087 | -0.0000 | 0.0002 |
| 500 | mi_unit_action | 1.4670 ± 0.0084 | 0.0624 | 0.0013 |
| 500 | mi_unit_subject | 1.7088 ± 0.0135 | 0.1191 | 0.0017 |
| 500 | cmi_unit_subject_given_action | 1.7376 ± 0.0108 | 0.1944 | 0.0022 |
| 500 | h_unit | 5.3199 ± 0.0188 | 5.3199 | 0.0000 |
| 500 | h_action_given_unit | 0.5590 ± 0.0084 | 1.9635 | 0.0013 |
| 500 | h_subject_given_unit | 1.1355 ± 0.0135 | 2.7252 | 0.0017 |
| 500 | nmi_unit_action | 0.3994 ± 0.0015 | 0.0170 | 0.0004 |
| 500 | nmi_unit_subject | 0.4186 ± 0.0025 | 0.0292 | 0.0004 |
| 500 | ami_unit_action | 0.3891 ± 0.0015 | 0.0001 | 0.0004 |
| 500 | ami_unit_subject | 0.4011 ± 0.0027 | -0.0001 | 0.0004 |
| 1000 | mi_unit_action | 1.5348 ± 0.0013 | 0.1186 | 0.0016 |
| 1000 | mi_unit_subject | 2.0314 ± 0.0049 | 0.2257 | 0.0018 |
| 1000 | cmi_unit_subject_given_action | 2.0370 ± 0.0026 | 0.3062 | 0.0021 |
| 1000 | h_unit | 5.8682 ± 0.0054 | 5.8682 | 0.0000 |
| 1000 | h_action_given_unit | 0.4911 ± 0.0013 | 1.9073 | 0.0016 |
| 1000 | h_subject_given_unit | 0.8130 ± 0.0049 | 2.6186 | 0.0018 |
| 1000 | nmi_unit_action | 0.3888 ± 0.0002 | 0.0301 | 0.0004 |
| 1000 | nmi_unit_subject | 0.4663 ± 0.0009 | 0.0518 | 0.0004 |
| 1000 | ami_unit_action | 0.3699 ± 0.0002 | 0.0000 | 0.0004 |
| 1000 | ami_unit_subject | 0.4371 ± 0.0010 | -0.0000 | 0.0004 |

## hugadb, latent, W=60

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.9846 ± 0.0042 | 0.0022 | 0.0003 |
| 10 | mi_unit_subject | 0.2038 ± 0.0128 | 0.0040 | 0.0005 |
| 10 | cmi_unit_subject_given_action | 0.2332 ± 0.0138 | 0.0157 | 0.0006 |
| 10 | h_unit | 1.9461 ± 0.0367 | 1.9461 | 0.0000 |
| 10 | h_action_given_unit | 1.0423 ± 0.0042 | 2.0247 | 0.0003 |
| 10 | h_subject_given_unit | 2.6409 ± 0.0128 | 2.8407 | 0.0005 |
| 10 | nmi_unit_action | 0.4957 ± 0.0066 | 0.0011 | 0.0002 |
| 10 | nmi_unit_subject | 0.0851 ± 0.0059 | 0.0017 | 0.0002 |
| 10 | ami_unit_action | 0.4952 ± 0.0066 | 0.0000 | 0.0002 |
| 10 | ami_unit_subject | 0.0836 ± 0.0059 | -0.0000 | 0.0002 |
| 20 | mi_unit_action | 1.0155 ± 0.0063 | 0.0046 | 0.0004 |
| 20 | mi_unit_subject | 0.2330 ± 0.0062 | 0.0085 | 0.0006 |
| 20 | cmi_unit_subject_given_action | 0.2769 ± 0.0205 | 0.0293 | 0.0009 |
| 20 | h_unit | 2.4222 ± 0.0367 | 2.4222 | 0.0000 |
| 20 | h_action_given_unit | 1.0114 ± 0.0063 | 2.0223 | 0.0004 |
| 20 | h_subject_given_unit | 2.6117 ± 0.0062 | 2.8362 | 0.0006 |
| 20 | nmi_unit_action | 0.4565 ± 0.0015 | 0.0021 | 0.0002 |
| 20 | nmi_unit_subject | 0.0885 ± 0.0017 | 0.0032 | 0.0002 |
| 20 | ami_unit_action | 0.4554 ± 0.0015 | 0.0000 | 0.0002 |
| 20 | ami_unit_subject | 0.0855 ± 0.0018 | -0.0000 | 0.0002 |
| 50 | mi_unit_action | 1.1575 ± 0.0080 | 0.0121 | 0.0007 |
| 50 | mi_unit_subject | 0.6355 ± 0.0122 | 0.0224 | 0.0012 |
| 50 | cmi_unit_subject_given_action | 0.6856 ± 0.0118 | 0.0625 | 0.0015 |
| 50 | h_unit | 3.3103 ± 0.0022 | 3.3103 | 0.0000 |
| 50 | h_action_given_unit | 0.8694 ± 0.0080 | 2.0148 | 0.0007 |
| 50 | h_subject_given_unit | 2.2092 ± 0.0122 | 2.8223 | 0.0012 |
| 50 | nmi_unit_action | 0.4338 ± 0.0029 | 0.0045 | 0.0003 |
| 50 | nmi_unit_subject | 0.2065 ± 0.0040 | 0.0073 | 0.0004 |
| 50 | ami_unit_action | 0.4312 ± 0.0029 | 0.0001 | 0.0003 |
| 50 | ami_unit_subject | 0.2007 ± 0.0040 | -0.0000 | 0.0004 |
| 100 | mi_unit_action | 1.2885 ± 0.0020 | 0.0246 | 0.0010 |
| 100 | mi_unit_subject | 0.9745 ± 0.0377 | 0.0466 | 0.0016 |
| 100 | cmi_unit_subject_given_action | 1.0294 ± 0.0409 | 0.1116 | 0.0021 |
| 100 | h_unit | 3.9628 ± 0.0530 | 3.9628 | 0.0000 |
| 100 | h_action_given_unit | 0.7383 ± 0.0020 | 2.0023 | 0.0010 |
| 100 | h_subject_given_unit | 1.8702 ± 0.0377 | 2.7981 | 0.0016 |
| 100 | nmi_unit_action | 0.4303 ± 0.0038 | 0.0082 | 0.0003 |
| 100 | nmi_unit_subject | 0.2862 ± 0.0088 | 0.0137 | 0.0005 |
| 100 | ami_unit_action | 0.4256 ± 0.0038 | 0.0001 | 0.0003 |
| 100 | ami_unit_subject | 0.2764 ± 0.0091 | 0.0000 | 0.0005 |
| 500 | mi_unit_action | 1.5095 ± 0.0059 | 0.1216 | 0.0020 |
| 500 | mi_unit_subject | 1.8102 ± 0.0041 | 0.2328 | 0.0025 |
| 500 | cmi_unit_subject_given_action | 1.8306 ± 0.0016 | 0.3299 | 0.0029 |
| 500 | h_unit | 5.3784 ± 0.0043 | 5.3784 | 0.0000 |
| 500 | h_action_given_unit | 0.5174 ± 0.0059 | 1.9053 | 0.0020 |
| 500 | h_subject_given_unit | 1.0345 ± 0.0041 | 2.6119 | 0.0025 |
| 500 | nmi_unit_action | 0.4077 ± 0.0014 | 0.0328 | 0.0005 |
| 500 | nmi_unit_subject | 0.4403 ± 0.0008 | 0.0566 | 0.0006 |
| 500 | ami_unit_action | 0.3876 ± 0.0015 | 0.0001 | 0.0006 |
| 500 | ami_unit_subject | 0.4066 ± 0.0010 | -0.0001 | 0.0006 |
| 1000 | mi_unit_action | 1.5691 ± 0.0033 | 0.2215 | 0.0024 |
| 1000 | mi_unit_subject | 2.0944 ± 0.0030 | 0.4088 | 0.0026 |
| 1000 | cmi_unit_subject_given_action | 2.0969 ± 0.0018 | 0.5008 | 0.0027 |
| 1000 | h_unit | 5.8895 ± 0.0032 | 5.8895 | 0.0000 |
| 1000 | h_action_given_unit | 0.4578 ± 0.0033 | 1.8054 | 0.0024 |
| 1000 | h_subject_given_unit | 0.7502 ± 0.0030 | 2.4359 | 0.0026 |
| 1000 | nmi_unit_action | 0.3964 ± 0.0007 | 0.0560 | 0.0006 |
| 1000 | nmi_unit_subject | 0.4796 ± 0.0005 | 0.0936 | 0.0006 |
| 1000 | ami_unit_action | 0.3607 ± 0.0007 | 0.0001 | 0.0006 |
| 1000 | ami_unit_subject | 0.4258 ± 0.0006 | -0.0001 | 0.0007 |

## hugadb, per_recording, W=15

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.6499 ± 0.0178 | 0.0005 | 0.0001 |
| 10 | mi_unit_subject | 0.0544 ± 0.0002 | 0.0010 | 0.0001 |
| 10 | cmi_unit_subject_given_action | 0.0919 ± 0.0013 | 0.0084 | 0.0003 |
| 10 | h_unit | 2.1280 ± 0.0537 | 2.1280 | 0.0000 |
| 10 | h_action_given_unit | 1.3777 ± 0.0178 | 2.0271 | 0.0001 |
| 10 | h_subject_given_unit | 2.7898 ± 0.0002 | 2.8432 | 0.0001 |
| 10 | nmi_unit_action | 0.3127 ± 0.0052 | 0.0003 | 0.0000 |
| 10 | nmi_unit_subject | 0.0219 ± 0.0002 | 0.0004 | 0.0000 |
| 10 | ami_unit_action | 0.3126 ± 0.0052 | -0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0215 ± 0.0002 | 0.0000 | 0.0000 |
| 100 | mi_unit_action | 1.2374 ± 0.0067 | 0.0058 | 0.0003 |
| 100 | mi_unit_subject | 0.3112 ± 0.0068 | 0.0108 | 0.0003 |
| 100 | cmi_unit_subject_given_action | 0.3914 ± 0.0116 | 0.0498 | 0.0005 |
| 100 | h_unit | 4.2670 ± 0.0249 | 4.2670 | 0.0000 |
| 100 | h_action_given_unit | 0.7903 ± 0.0067 | 2.0219 | 0.0003 |
| 100 | h_subject_given_unit | 2.5330 ± 0.0068 | 2.8334 | 0.0003 |
| 100 | nmi_unit_action | 0.3932 ± 0.0030 | 0.0018 | 0.0001 |
| 100 | nmi_unit_subject | 0.0875 ± 0.0017 | 0.0030 | 0.0001 |
| 100 | ami_unit_action | 0.3920 ± 0.0030 | -0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.0847 ± 0.0017 | -0.0000 | 0.0001 |
| 1000 | mi_unit_action | 1.5263 ± 0.0026 | 0.0596 | 0.0008 |
| 1000 | mi_unit_subject | 1.1651 ± 0.0041 | 0.1127 | 0.0010 |
| 1000 | cmi_unit_subject_given_action | 1.2505 ± 0.0045 | 0.2213 | 0.0010 |
| 1000 | h_unit | 6.4014 ± 0.0020 | 6.4014 | 0.0000 |
| 1000 | h_action_given_unit | 0.5014 ± 0.0026 | 1.9680 | 0.0008 |
| 1000 | h_subject_given_unit | 1.6792 ± 0.0041 | 2.7315 | 0.0010 |
| 1000 | nmi_unit_action | 0.3621 ± 0.0007 | 0.0141 | 0.0002 |
| 1000 | nmi_unit_subject | 0.2520 ± 0.0009 | 0.0244 | 0.0002 |
| 1000 | ami_unit_action | 0.3530 ± 0.0007 | -0.0000 | 0.0002 |
| 1000 | ami_unit_subject | 0.2333 ± 0.0008 | -0.0001 | 0.0002 |

## hugadb, per_recording, W=60

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.7699 ± 0.0284 | 0.0021 | 0.0004 |
| 10 | mi_unit_subject | 0.0531 ± 0.0086 | 0.0040 | 0.0004 |
| 10 | cmi_unit_subject_given_action | 0.1079 ± 0.0052 | 0.0263 | 0.0011 |
| 10 | h_unit | 2.2100 ± 0.0209 | 2.2100 | 0.0000 |
| 10 | h_action_given_unit | 1.2570 ± 0.0284 | 2.0248 | 0.0004 |
| 10 | h_subject_given_unit | 2.7916 ± 0.0086 | 2.8407 | 0.0004 |
| 10 | nmi_unit_action | 0.3634 ± 0.0127 | 0.0010 | 0.0002 |
| 10 | nmi_unit_subject | 0.0210 ± 0.0034 | 0.0016 | 0.0002 |
| 10 | ami_unit_action | 0.3628 ± 0.0127 | -0.0000 | 0.0002 |
| 10 | ami_unit_subject | 0.0194 ± 0.0034 | -0.0000 | 0.0002 |
| 100 | mi_unit_action | 1.3459 ± 0.0023 | 0.0231 | 0.0009 |
| 100 | mi_unit_subject | 0.3365 ± 0.0072 | 0.0430 | 0.0013 |
| 100 | cmi_unit_subject_given_action | 0.4316 ± 0.0069 | 0.1337 | 0.0017 |
| 100 | h_unit | 4.2616 ± 0.0133 | 4.2616 | 0.0000 |
| 100 | h_action_given_unit | 0.6810 ± 0.0023 | 2.0038 | 0.0009 |
| 100 | h_subject_given_unit | 2.5082 ± 0.0072 | 2.8017 | 0.0013 |
| 100 | nmi_unit_action | 0.4281 ± 0.0015 | 0.0074 | 0.0003 |
| 100 | nmi_unit_subject | 0.0947 ± 0.0018 | 0.0121 | 0.0004 |
| 100 | ami_unit_action | 0.4238 ± 0.0015 | 0.0000 | 0.0003 |
| 100 | ami_unit_subject | 0.0836 ± 0.0019 | -0.0000 | 0.0004 |
| 1000 | mi_unit_action | 1.6258 ± 0.0007 | 0.2246 | 0.0030 |
| 1000 | mi_unit_subject | 1.3623 ± 0.0149 | 0.4225 | 0.0038 |
| 1000 | cmi_unit_subject_given_action | 1.4411 ± 0.0147 | 0.5674 | 0.0039 |
| 1000 | h_unit | 6.4281 ± 0.0089 | 6.4281 | 0.0000 |
| 1000 | h_action_given_unit | 0.4011 ± 0.0007 | 1.8023 | 0.0030 |
| 1000 | h_subject_given_unit | 1.4824 ± 0.0149 | 2.4222 | 0.0038 |
| 1000 | nmi_unit_action | 0.3846 ± 0.0004 | 0.0531 | 0.0007 |
| 1000 | nmi_unit_subject | 0.2938 ± 0.0029 | 0.0911 | 0.0008 |
| 1000 | ami_unit_action | 0.3501 ± 0.0004 | 0.0001 | 0.0008 |
| 1000 | ami_unit_subject | 0.2230 ± 0.0032 | -0.0000 | 0.0009 |

## hugadb, raw, W=15

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.6185 ± 0.0029 | 0.0005 | 0.0001 |
| 10 | mi_unit_subject | 0.0831 ± 0.0027 | 0.0010 | 0.0001 |
| 10 | cmi_unit_subject_given_action | 0.1400 ± 0.0043 | 0.0042 | 0.0002 |
| 10 | h_unit | 1.8216 ± 0.0060 | 1.8216 | 0.0000 |
| 10 | h_action_given_unit | 1.4091 ± 0.0029 | 2.0271 | 0.0001 |
| 10 | h_subject_given_unit | 2.7611 ± 0.0027 | 2.8432 | 0.0001 |
| 10 | nmi_unit_action | 0.3214 ± 0.0010 | 0.0003 | 0.0000 |
| 10 | nmi_unit_subject | 0.0356 ± 0.0012 | 0.0004 | 0.0001 |
| 10 | ami_unit_action | 0.3212 ± 0.0010 | -0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0352 ± 0.0012 | -0.0000 | 0.0001 |
| 100 | mi_unit_action | 1.1555 ± 0.0081 | 0.0059 | 0.0003 |
| 100 | mi_unit_subject | 0.7952 ± 0.0783 | 0.0112 | 0.0003 |
| 100 | cmi_unit_subject_given_action | 0.8115 ± 0.0830 | 0.0329 | 0.0006 |
| 100 | h_unit | 3.7481 ± 0.0991 | 3.7481 | 0.0000 |
| 100 | h_action_given_unit | 0.8721 ± 0.0081 | 2.0217 | 0.0003 |
| 100 | h_subject_given_unit | 2.0490 ± 0.0783 | 2.8330 | 0.0003 |
| 100 | nmi_unit_action | 0.4002 ± 0.0056 | 0.0021 | 0.0001 |
| 100 | nmi_unit_subject | 0.2410 ± 0.0199 | 0.0034 | 0.0001 |
| 100 | ami_unit_action | 0.3990 ± 0.0056 | -0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.2383 ± 0.0200 | -0.0000 | 0.0001 |
| 1000 | mi_unit_action | 1.4647 ± 0.0037 | 0.0612 | 0.0008 |
| 1000 | mi_unit_subject | 1.9308 ± 0.0112 | 0.1176 | 0.0007 |
| 1000 | cmi_unit_subject_given_action | 1.9182 ± 0.0090 | 0.1778 | 0.0014 |
| 1000 | h_unit | 5.7187 ± 0.0226 | 5.7187 | 0.0000 |
| 1000 | h_action_given_unit | 0.5630 ± 0.0037 | 1.9665 | 0.0008 |
| 1000 | h_subject_given_unit | 0.9134 ± 0.0112 | 2.7267 | 0.0007 |
| 1000 | nmi_unit_action | 0.3782 ± 0.0003 | 0.0158 | 0.0002 |
| 1000 | nmi_unit_subject | 0.4510 ± 0.0016 | 0.0275 | 0.0002 |
| 1000 | ami_unit_action | 0.3682 ± 0.0002 | -0.0000 | 0.0002 |
| 1000 | ami_unit_subject | 0.4355 ± 0.0016 | 0.0000 | 0.0002 |

## hugadb, raw, W=60

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.7331 ± 0.0084 | 0.0022 | 0.0003 |
| 10 | mi_unit_subject | 0.0635 ± 0.0025 | 0.0041 | 0.0005 |
| 10 | cmi_unit_subject_given_action | 0.1158 ± 0.0026 | 0.0158 | 0.0006 |
| 10 | h_unit | 1.7952 ± 0.0251 | 1.7952 | 0.0000 |
| 10 | h_action_given_unit | 1.2938 ± 0.0084 | 2.0247 | 0.0003 |
| 10 | h_subject_given_unit | 2.7812 ± 0.0025 | 2.8406 | 0.0005 |
| 10 | nmi_unit_action | 0.3837 ± 0.0069 | 0.0011 | 0.0002 |
| 10 | nmi_unit_subject | 0.0274 ± 0.0012 | 0.0018 | 0.0002 |
| 10 | ami_unit_action | 0.3830 ± 0.0069 | 0.0000 | 0.0002 |
| 10 | ami_unit_subject | 0.0257 ± 0.0012 | 0.0000 | 0.0002 |
| 100 | mi_unit_action | 1.1645 ± 0.0169 | 0.0244 | 0.0012 |
| 100 | mi_unit_subject | 0.7868 ± 0.0448 | 0.0470 | 0.0017 |
| 100 | cmi_unit_subject_given_action | 0.8360 ± 0.0516 | 0.1068 | 0.0016 |
| 100 | h_unit | 3.7197 ± 0.0984 | 3.7197 | 0.0000 |
| 100 | h_action_given_unit | 0.8624 ± 0.0169 | 2.0025 | 0.0012 |
| 100 | h_subject_given_unit | 2.0578 ± 0.0448 | 2.7977 | 0.0017 |
| 100 | nmi_unit_action | 0.4053 ± 0.0050 | 0.0085 | 0.0004 |
| 100 | nmi_unit_subject | 0.2396 ± 0.0104 | 0.0143 | 0.0005 |
| 100 | ami_unit_action | 0.4002 ± 0.0050 | -0.0000 | 0.0004 |
| 100 | ami_unit_subject | 0.2286 ± 0.0107 | 0.0001 | 0.0005 |
| 1000 | mi_unit_action | 1.4906 ± 0.0060 | 0.2184 | 0.0026 |
| 1000 | mi_unit_subject | 2.0613 ± 0.0228 | 0.3994 | 0.0026 |
| 1000 | cmi_unit_subject_given_action | 2.0627 ± 0.0200 | 0.5049 | 0.0028 |
| 1000 | h_unit | 5.7857 ± 0.0285 | 5.7857 | 0.0000 |
| 1000 | h_action_given_unit | 0.5363 ± 0.0060 | 1.8085 | 0.0026 |
| 1000 | h_subject_given_unit | 0.7834 ± 0.0228 | 2.4453 | 0.0026 |
| 1000 | nmi_unit_action | 0.3816 ± 0.0008 | 0.0559 | 0.0007 |
| 1000 | nmi_unit_subject | 0.4777 ± 0.0037 | 0.0926 | 0.0006 |
| 1000 | ami_unit_action | 0.3450 ± 0.0008 | 0.0001 | 0.0007 |
| 1000 | ami_unit_subject | 0.4244 ± 0.0043 | 0.0001 | 0.0007 |

## lara, latent, W=12

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.2827 ± 0.0061 | 0.0001 | 0.0000 |
| 10 | mi_unit_subject | 0.5764 ± 0.0312 | 0.0003 | 0.0000 |
| 10 | cmi_unit_subject_given_action | 0.6260 ± 0.0410 | 0.0022 | 0.0001 |
| 10 | h_unit | 2.0049 ± 0.0571 | 2.0049 | 0.0000 |
| 10 | h_action_given_unit | 1.4290 ± 0.0061 | 1.7115 | 0.0000 |
| 10 | h_subject_given_unit | 2.1823 ± 0.0312 | 2.7583 | 0.0000 |
| 10 | nmi_unit_action | 0.1521 ± 0.0009 | 0.0001 | 0.0000 |
| 10 | nmi_unit_subject | 0.2419 ± 0.0101 | 0.0001 | 0.0000 |
| 10 | ami_unit_action | 0.1520 ± 0.0009 | 0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.2418 ± 0.0101 | -0.0000 | 0.0000 |
| 20 | mi_unit_action | 0.3431 ± 0.0093 | 0.0003 | 0.0000 |
| 20 | mi_unit_subject | 0.8797 ± 0.0650 | 0.0006 | 0.0000 |
| 20 | cmi_unit_subject_given_action | 0.9383 ± 0.0598 | 0.0047 | 0.0002 |
| 20 | h_unit | 2.7477 ± 0.0638 | 2.7477 | 0.0000 |
| 20 | h_action_given_unit | 1.3686 ± 0.0093 | 1.7114 | 0.0000 |
| 20 | h_subject_given_unit | 1.8789 ± 0.0650 | 2.7580 | 0.0000 |
| 20 | nmi_unit_action | 0.1539 ± 0.0051 | 0.0001 | 0.0000 |
| 20 | nmi_unit_subject | 0.3193 ± 0.0201 | 0.0002 | 0.0000 |
| 20 | ami_unit_action | 0.1538 ± 0.0051 | 0.0000 | 0.0000 |
| 20 | ami_unit_subject | 0.3191 ± 0.0201 | -0.0000 | 0.0000 |
| 50 | mi_unit_action | 0.4132 ± 0.0059 | 0.0008 | 0.0000 |
| 50 | mi_unit_subject | 1.1518 ± 0.0304 | 0.0017 | 0.0001 |
| 50 | cmi_unit_subject_given_action | 1.2066 ± 0.0284 | 0.0113 | 0.0002 |
| 50 | h_unit | 3.6115 ± 0.0256 | 3.6115 | 0.0000 |
| 50 | h_action_given_unit | 1.2985 ± 0.0059 | 1.7109 | 0.0000 |
| 50 | h_subject_given_unit | 1.6068 ± 0.0304 | 2.7570 | 0.0001 |
| 50 | nmi_unit_action | 0.1552 ± 0.0020 | 0.0003 | 0.0000 |
| 50 | nmi_unit_subject | 0.3616 ± 0.0082 | 0.0005 | 0.0000 |
| 50 | ami_unit_action | 0.1550 ± 0.0020 | 0.0000 | 0.0000 |
| 50 | ami_unit_subject | 0.3613 ± 0.0082 | -0.0000 | 0.0000 |
| 100 | mi_unit_action | 0.4824 ± 0.0038 | 0.0016 | 0.0001 |
| 100 | mi_unit_subject | 1.3659 ± 0.0125 | 0.0033 | 0.0001 |
| 100 | cmi_unit_subject_given_action | 1.4201 ± 0.0124 | 0.0221 | 0.0003 |
| 100 | h_unit | 4.3349 ± 0.0200 | 4.3349 | 0.0000 |
| 100 | h_action_given_unit | 1.2293 ± 0.0038 | 1.7101 | 0.0001 |
| 100 | h_subject_given_unit | 1.3927 ± 0.0125 | 2.7553 | 0.0001 |
| 100 | nmi_unit_action | 0.1596 ± 0.0007 | 0.0005 | 0.0000 |
| 100 | nmi_unit_subject | 0.3851 ± 0.0025 | 0.0009 | 0.0000 |
| 100 | ami_unit_action | 0.1591 ± 0.0007 | 0.0000 | 0.0000 |
| 100 | ami_unit_subject | 0.3845 ± 0.0025 | -0.0000 | 0.0000 |
| 500 | mi_unit_action | 0.6466 ± 0.0027 | 0.0082 | 0.0002 |
| 500 | mi_unit_subject | 1.8116 ± 0.0048 | 0.0174 | 0.0002 |
| 500 | cmi_unit_subject_given_action | 1.8625 ± 0.0028 | 0.0874 | 0.0005 |
| 500 | h_unit | 5.9147 ± 0.0109 | 5.9147 | 0.0000 |
| 500 | h_action_given_unit | 1.0651 ± 0.0027 | 1.7035 | 0.0002 |
| 500 | h_subject_given_unit | 0.9471 ± 0.0048 | 2.7412 | 0.0002 |
| 500 | nmi_unit_action | 0.1696 ± 0.0006 | 0.0021 | 0.0001 |
| 500 | nmi_unit_subject | 0.4177 ± 0.0010 | 0.0040 | 0.0001 |
| 500 | ami_unit_action | 0.1678 ± 0.0006 | 0.0000 | 0.0001 |
| 500 | ami_unit_subject | 0.4154 ± 0.0010 | -0.0000 | 0.0001 |
| 1000 | mi_unit_action | 0.7233 ± 0.0028 | 0.0165 | 0.0003 |
| 1000 | mi_unit_subject | 1.9837 ± 0.0044 | 0.0354 | 0.0004 |
| 1000 | cmi_unit_subject_given_action | 2.0264 ± 0.0030 | 0.1461 | 0.0006 |
| 1000 | h_unit | 6.5607 ± 0.0036 | 6.5607 | 0.0000 |
| 1000 | h_action_given_unit | 0.9884 ± 0.0028 | 1.6952 | 0.0003 |
| 1000 | h_subject_given_unit | 0.7749 ± 0.0044 | 2.7233 | 0.0004 |
| 1000 | nmi_unit_action | 0.1749 ± 0.0006 | 0.0040 | 0.0001 |
| 1000 | nmi_unit_subject | 0.4257 ± 0.0009 | 0.0076 | 0.0001 |
| 1000 | ami_unit_action | 0.1716 ± 0.0006 | 0.0000 | 0.0001 |
| 1000 | ami_unit_subject | 0.4213 ± 0.0009 | -0.0000 | 0.0001 |

## lara, latent, W=25

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.2899 ± 0.0076 | 0.0003 | 0.0001 |
| 10 | mi_unit_subject | 0.5584 ± 0.0376 | 0.0006 | 0.0001 |
| 10 | cmi_unit_subject_given_action | 0.6138 ± 0.0420 | 0.0047 | 0.0002 |
| 10 | h_unit | 2.0365 ± 0.0496 | 2.0365 | 0.0000 |
| 10 | h_action_given_unit | 1.4196 ± 0.0076 | 1.7092 | 0.0001 |
| 10 | h_subject_given_unit | 2.2002 ± 0.0376 | 2.7580 | 0.0001 |
| 10 | nmi_unit_action | 0.1548 ± 0.0020 | 0.0002 | 0.0000 |
| 10 | nmi_unit_subject | 0.2329 ± 0.0146 | 0.0003 | 0.0000 |
| 10 | ami_unit_action | 0.1546 ± 0.0020 | -0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.2326 ± 0.0146 | -0.0000 | 0.0000 |
| 20 | mi_unit_action | 0.3528 ± 0.0024 | 0.0006 | 0.0001 |
| 20 | mi_unit_subject | 0.8190 ± 0.0305 | 0.0013 | 0.0001 |
| 20 | cmi_unit_subject_given_action | 0.8789 ± 0.0349 | 0.0096 | 0.0003 |
| 20 | h_unit | 2.7282 ± 0.0453 | 2.7282 | 0.0000 |
| 20 | h_action_given_unit | 1.3567 ± 0.0024 | 1.7089 | 0.0001 |
| 20 | h_subject_given_unit | 1.9397 ± 0.0305 | 2.7573 | 0.0001 |
| 20 | nmi_unit_action | 0.1590 ± 0.0027 | 0.0003 | 0.0000 |
| 20 | nmi_unit_subject | 0.2985 ± 0.0098 | 0.0005 | 0.0000 |
| 20 | ami_unit_action | 0.1588 ± 0.0027 | -0.0000 | 0.0000 |
| 20 | ami_unit_subject | 0.2981 ± 0.0098 | -0.0000 | 0.0000 |
| 50 | mi_unit_action | 0.4202 ± 0.0031 | 0.0016 | 0.0002 |
| 50 | mi_unit_subject | 1.1464 ± 0.0085 | 0.0036 | 0.0002 |
| 50 | cmi_unit_subject_given_action | 1.2049 ± 0.0076 | 0.0229 | 0.0004 |
| 50 | h_unit | 3.6156 ± 0.0146 | 3.6156 | 0.0000 |
| 50 | h_action_given_unit | 1.2893 ± 0.0031 | 1.7079 | 0.0002 |
| 50 | h_subject_given_unit | 1.6123 ± 0.0085 | 2.7550 | 0.0002 |
| 50 | nmi_unit_action | 0.1578 ± 0.0008 | 0.0006 | 0.0001 |
| 50 | nmi_unit_subject | 0.3597 ± 0.0034 | 0.0011 | 0.0001 |
| 50 | ami_unit_action | 0.1573 ± 0.0008 | -0.0000 | 0.0001 |
| 50 | ami_unit_subject | 0.3590 ± 0.0034 | 0.0000 | 0.0001 |
| 100 | mi_unit_action | 0.4855 ± 0.0048 | 0.0033 | 0.0002 |
| 100 | mi_unit_subject | 1.3807 ± 0.0160 | 0.0072 | 0.0002 |
| 100 | cmi_unit_subject_given_action | 1.4399 ± 0.0150 | 0.0441 | 0.0005 |
| 100 | h_unit | 4.3429 ± 0.0111 | 4.3429 | 0.0000 |
| 100 | h_action_given_unit | 1.2240 ± 0.0048 | 1.7062 | 0.0002 |
| 100 | h_subject_given_unit | 1.3779 ± 0.0160 | 2.7514 | 0.0002 |
| 100 | nmi_unit_action | 0.1604 ± 0.0017 | 0.0011 | 0.0001 |
| 100 | nmi_unit_subject | 0.3889 ± 0.0048 | 0.0020 | 0.0001 |
| 100 | ami_unit_action | 0.1595 ± 0.0017 | -0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.3876 ± 0.0048 | 0.0000 | 0.0001 |
| 500 | mi_unit_action | 0.6668 ± 0.0020 | 0.0172 | 0.0004 |
| 500 | mi_unit_subject | 1.8230 ± 0.0043 | 0.0369 | 0.0005 |
| 500 | cmi_unit_subject_given_action | 1.8787 ± 0.0034 | 0.1575 | 0.0010 |
| 500 | h_unit | 5.8925 ± 0.0087 | 5.8925 | 0.0000 |
| 500 | h_action_given_unit | 1.0427 ± 0.0020 | 1.6923 | 0.0004 |
| 500 | h_subject_given_unit | 0.9356 ± 0.0043 | 2.7217 | 0.0005 |
| 500 | nmi_unit_action | 0.1754 ± 0.0004 | 0.0045 | 0.0001 |
| 500 | nmi_unit_subject | 0.4214 ± 0.0014 | 0.0085 | 0.0001 |
| 500 | ami_unit_action | 0.1717 ± 0.0004 | 0.0000 | 0.0001 |
| 500 | ami_unit_subject | 0.4165 ± 0.0014 | 0.0000 | 0.0001 |
| 1000 | mi_unit_action | 0.7519 ± 0.0028 | 0.0346 | 0.0006 |
| 1000 | mi_unit_subject | 2.0061 ± 0.0135 | 0.0746 | 0.0007 |
| 1000 | cmi_unit_subject_given_action | 2.0529 ± 0.0116 | 0.2560 | 0.0013 |
| 1000 | h_unit | 6.5450 ± 0.0051 | 6.5450 | 0.0000 |
| 1000 | h_action_given_unit | 0.9576 ± 0.0028 | 1.6749 | 0.0006 |
| 1000 | h_subject_given_unit | 0.7526 ± 0.0135 | 2.6840 | 0.0007 |
| 1000 | nmi_unit_action | 0.1822 ± 0.0007 | 0.0084 | 0.0001 |
| 1000 | nmi_unit_subject | 0.4312 ± 0.0029 | 0.0160 | 0.0002 |
| 1000 | ami_unit_action | 0.1753 ± 0.0007 | 0.0000 | 0.0001 |
| 1000 | ami_unit_subject | 0.4220 ± 0.0029 | 0.0000 | 0.0002 |

## lara, latent, W=50

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.2932 ± 0.0068 | 0.0006 | 0.0001 |
| 10 | mi_unit_subject | 0.6059 ± 0.0347 | 0.0013 | 0.0002 |
| 10 | cmi_unit_subject_given_action | 0.6652 ± 0.0445 | 0.0090 | 0.0004 |
| 10 | h_unit | 2.0516 ± 0.0555 | 2.0516 | 0.0000 |
| 10 | h_action_given_unit | 1.4174 ± 0.0068 | 1.7100 | 0.0001 |
| 10 | h_subject_given_unit | 2.1527 ± 0.0347 | 2.7573 | 0.0002 |
| 10 | nmi_unit_action | 0.1558 ± 0.0013 | 0.0003 | 0.0001 |
| 10 | nmi_unit_subject | 0.2518 ± 0.0116 | 0.0005 | 0.0001 |
| 10 | ami_unit_action | 0.1556 ± 0.0014 | 0.0000 | 0.0001 |
| 10 | ami_unit_subject | 0.2514 ± 0.0116 | 0.0000 | 0.0001 |
| 20 | mi_unit_action | 0.3454 ± 0.0060 | 0.0013 | 0.0002 |
| 20 | mi_unit_subject | 0.8714 ± 0.0085 | 0.0027 | 0.0003 |
| 20 | cmi_unit_subject_given_action | 0.9363 ± 0.0028 | 0.0181 | 0.0006 |
| 20 | h_unit | 2.7190 ± 0.0346 | 2.7190 | 0.0000 |
| 20 | h_action_given_unit | 1.3652 ± 0.0060 | 1.7093 | 0.0002 |
| 20 | h_subject_given_unit | 1.8873 ± 0.0085 | 2.7559 | 0.0003 |
| 20 | nmi_unit_action | 0.1559 ± 0.0018 | 0.0006 | 0.0001 |
| 20 | nmi_unit_subject | 0.3182 ± 0.0049 | 0.0010 | 0.0001 |
| 20 | ami_unit_action | 0.1555 ± 0.0018 | 0.0000 | 0.0001 |
| 20 | ami_unit_subject | 0.3175 ± 0.0049 | 0.0000 | 0.0001 |
| 50 | mi_unit_action | 0.4321 ± 0.0038 | 0.0033 | 0.0002 |
| 50 | mi_unit_subject | 1.1777 ± 0.0125 | 0.0071 | 0.0004 |
| 50 | cmi_unit_subject_given_action | 1.2387 ± 0.0137 | 0.0437 | 0.0007 |
| 50 | h_unit | 3.6397 ± 0.0118 | 3.6397 | 0.0000 |
| 50 | h_action_given_unit | 1.2785 ± 0.0038 | 1.7073 | 0.0002 |
| 50 | h_subject_given_unit | 1.5809 ± 0.0125 | 2.7516 | 0.0004 |
| 50 | nmi_unit_action | 0.1615 ± 0.0018 | 0.0012 | 0.0001 |
| 50 | nmi_unit_subject | 0.3681 ± 0.0041 | 0.0022 | 0.0001 |
| 50 | ami_unit_action | 0.1605 ± 0.0018 | 0.0000 | 0.0001 |
| 50 | ami_unit_subject | 0.3667 ± 0.0041 | 0.0000 | 0.0001 |
| 100 | mi_unit_action | 0.5246 ± 0.0031 | 0.0067 | 0.0003 |
| 100 | mi_unit_subject | 1.3948 ± 0.0036 | 0.0145 | 0.0005 |
| 100 | cmi_unit_subject_given_action | 1.4585 ± 0.0053 | 0.0804 | 0.0010 |
| 100 | h_unit | 4.3449 ± 0.0271 | 4.3449 | 0.0000 |
| 100 | h_action_given_unit | 1.1860 ± 0.0031 | 1.7039 | 0.0003 |
| 100 | h_subject_given_unit | 1.3639 ± 0.0036 | 2.7442 | 0.0005 |
| 100 | nmi_unit_action | 0.1733 ± 0.0015 | 0.0022 | 0.0001 |
| 100 | nmi_unit_subject | 0.3927 ± 0.0015 | 0.0041 | 0.0002 |
| 100 | ami_unit_action | 0.1714 ± 0.0015 | 0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.3902 ± 0.0015 | 0.0000 | 0.0002 |
| 500 | mi_unit_action | 0.7070 ± 0.0015 | 0.0343 | 0.0008 |
| 500 | mi_unit_subject | 1.8643 ± 0.0064 | 0.0745 | 0.0014 |
| 500 | cmi_unit_subject_given_action | 1.9261 ± 0.0053 | 0.2652 | 0.0017 |
| 500 | h_unit | 5.8859 ± 0.0184 | 5.8859 | 0.0000 |
| 500 | h_action_given_unit | 1.0036 ± 0.0015 | 1.6763 | 0.0008 |
| 500 | h_subject_given_unit | 0.8944 ± 0.0064 | 2.6841 | 0.0014 |
| 500 | nmi_unit_action | 0.1861 ± 0.0005 | 0.0090 | 0.0002 |
| 500 | nmi_unit_subject | 0.4313 ± 0.0007 | 0.0172 | 0.0003 |
| 500 | ami_unit_action | 0.1787 ± 0.0005 | -0.0000 | 0.0002 |
| 500 | ami_unit_subject | 0.4214 ± 0.0007 | 0.0001 | 0.0003 |
| 1000 | mi_unit_action | 0.7983 ± 0.0031 | 0.0685 | 0.0012 |
| 1000 | mi_unit_subject | 2.0290 ± 0.0061 | 0.1481 | 0.0014 |
| 1000 | cmi_unit_subject_given_action | 2.0853 ± 0.0035 | 0.4115 | 0.0014 |
| 1000 | h_unit | 6.5165 ± 0.0098 | 6.5165 | 0.0000 |
| 1000 | h_action_given_unit | 0.9123 ± 0.0031 | 1.6421 | 0.0012 |
| 1000 | h_subject_given_unit | 0.7296 ± 0.0061 | 2.6105 | 0.0014 |
| 1000 | nmi_unit_action | 0.1941 ± 0.0006 | 0.0167 | 0.0003 |
| 1000 | nmi_unit_subject | 0.4375 ± 0.0010 | 0.0319 | 0.0003 |
| 1000 | ami_unit_action | 0.1804 ± 0.0006 | 0.0000 | 0.0003 |
| 1000 | ami_unit_subject | 0.4190 ± 0.0010 | 0.0001 | 0.0003 |

## lara, per_recording, W=12

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.2940 ± 0.0002 | 0.0002 | 0.0000 |
| 10 | mi_unit_subject | 0.0270 ± 0.0001 | 0.0003 | 0.0000 |
| 10 | cmi_unit_subject_given_action | 0.0647 ± 0.0001 | 0.0023 | 0.0001 |
| 10 | h_unit | 2.0537 ± 0.0016 | 2.0537 | 0.0000 |
| 10 | h_action_given_unit | 1.4176 ± 0.0002 | 1.7115 | 0.0000 |
| 10 | h_subject_given_unit | 2.7316 ± 0.0001 | 2.7583 | 0.0000 |
| 10 | nmi_unit_action | 0.1562 ± 0.0001 | 0.0001 | 0.0000 |
| 10 | nmi_unit_subject | 0.0112 ± 0.0001 | 0.0001 | 0.0000 |
| 10 | ami_unit_action | 0.1561 ± 0.0001 | 0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0111 ± 0.0001 | 0.0000 | 0.0000 |
| 100 | mi_unit_action | 0.4700 ± 0.0024 | 0.0016 | 0.0001 |
| 100 | mi_unit_subject | 0.1768 ± 0.0054 | 0.0034 | 0.0001 |
| 100 | cmi_unit_subject_given_action | 0.2837 ± 0.0059 | 0.0230 | 0.0003 |
| 100 | h_unit | 4.3500 ± 0.0063 | 4.3500 | 0.0000 |
| 100 | h_action_given_unit | 1.2416 ± 0.0024 | 1.7101 | 0.0001 |
| 100 | h_subject_given_unit | 2.5819 ± 0.0054 | 2.7553 | 0.0001 |
| 100 | nmi_unit_action | 0.1551 ± 0.0007 | 0.0005 | 0.0000 |
| 100 | nmi_unit_subject | 0.0497 ± 0.0015 | 0.0009 | 0.0000 |
| 100 | ami_unit_action | 0.1546 ± 0.0007 | 0.0000 | 0.0000 |
| 100 | ami_unit_subject | 0.0488 ± 0.0015 | -0.0000 | 0.0000 |
| 1000 | mi_unit_action | 0.6134 ± 0.0006 | 0.0164 | 0.0003 |
| 1000 | mi_unit_subject | 0.5300 ± 0.0058 | 0.0352 | 0.0003 |
| 1000 | cmi_unit_subject_given_action | 0.7579 ± 0.0032 | 0.1451 | 0.0006 |
| 1000 | h_unit | 6.3959 ± 0.0023 | 6.3959 | 0.0000 |
| 1000 | h_action_given_unit | 1.0983 ± 0.0006 | 1.6953 | 0.0003 |
| 1000 | h_subject_given_unit | 2.2286 ± 0.0058 | 2.7235 | 0.0003 |
| 1000 | nmi_unit_action | 0.1513 ± 0.0001 | 0.0040 | 0.0001 |
| 1000 | nmi_unit_subject | 0.1158 ± 0.0013 | 0.0077 | 0.0001 |
| 1000 | ami_unit_action | 0.1479 ± 0.0001 | 0.0000 | 0.0001 |
| 1000 | ami_unit_subject | 0.1089 ± 0.0013 | -0.0000 | 0.0001 |

## lara, per_recording, W=50

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.3079 ± 0.0005 | 0.0006 | 0.0001 |
| 10 | mi_unit_subject | 0.0271 ± 0.0004 | 0.0013 | 0.0002 |
| 10 | cmi_unit_subject_given_action | 0.0707 ± 0.0004 | 0.0094 | 0.0004 |
| 10 | h_unit | 2.0995 ± 0.0012 | 2.0995 | 0.0000 |
| 10 | h_action_given_unit | 1.4027 ± 0.0005 | 1.7100 | 0.0001 |
| 10 | h_subject_given_unit | 2.7315 ± 0.0004 | 2.7573 | 0.0002 |
| 10 | nmi_unit_action | 0.1616 ± 0.0002 | 0.0003 | 0.0000 |
| 10 | nmi_unit_subject | 0.0112 ± 0.0002 | 0.0005 | 0.0001 |
| 10 | ami_unit_action | 0.1614 ± 0.0002 | -0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0106 ± 0.0002 | 0.0000 | 0.0001 |
| 100 | mi_unit_action | 0.5256 ± 0.0012 | 0.0067 | 0.0004 |
| 100 | mi_unit_subject | 0.1986 ± 0.0045 | 0.0142 | 0.0005 |
| 100 | cmi_unit_subject_given_action | 0.3355 ± 0.0045 | 0.0798 | 0.0008 |
| 100 | h_unit | 4.2870 ± 0.0158 | 4.2870 | 0.0000 |
| 100 | h_action_given_unit | 1.1849 ± 0.0012 | 1.7039 | 0.0004 |
| 100 | h_subject_given_unit | 2.5600 ± 0.0045 | 2.7445 | 0.0005 |
| 100 | nmi_unit_action | 0.1753 ± 0.0005 | 0.0022 | 0.0001 |
| 100 | nmi_unit_subject | 0.0564 ± 0.0012 | 0.0040 | 0.0001 |
| 100 | ami_unit_action | 0.1735 ± 0.0005 | 0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.0525 ± 0.0011 | -0.0000 | 0.0001 |
| 1000 | mi_unit_action | 0.7112 ± 0.0021 | 0.0657 | 0.0010 |
| 1000 | mi_unit_subject | 0.6033 ± 0.0050 | 0.1395 | 0.0013 |
| 1000 | cmi_unit_subject_given_action | 0.8871 ± 0.0038 | 0.3931 | 0.0017 |
| 1000 | h_unit | 6.3362 ± 0.0101 | 6.3362 | 0.0000 |
| 1000 | h_action_given_unit | 0.9994 ± 0.0021 | 1.6449 | 0.0010 |
| 1000 | h_subject_given_unit | 2.1554 ± 0.0050 | 2.6191 | 0.0013 |
| 1000 | nmi_unit_action | 0.1768 ± 0.0003 | 0.0163 | 0.0002 |
| 1000 | nmi_unit_subject | 0.1327 ± 0.0010 | 0.0307 | 0.0003 |
| 1000 | ami_unit_action | 0.1631 ± 0.0003 | 0.0000 | 0.0002 |
| 1000 | ami_unit_subject | 0.1052 ± 0.0009 | 0.0000 | 0.0003 |

## lara, raw, W=12

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.1598 ± 0.0001 | 0.0001 | 0.0000 |
| 10 | mi_unit_subject | 0.1181 ± 0.0002 | 0.0003 | 0.0000 |
| 10 | cmi_unit_subject_given_action | 0.1719 ± 0.0001 | 0.0023 | 0.0001 |
| 10 | h_unit | 2.1506 ± 0.0010 | 2.1506 | 0.0000 |
| 10 | h_action_given_unit | 1.5519 ± 0.0001 | 1.7115 | 0.0000 |
| 10 | h_subject_given_unit | 2.6405 ± 0.0002 | 2.7583 | 0.0000 |
| 10 | nmi_unit_action | 0.0827 ± 0.0001 | 0.0001 | 0.0000 |
| 10 | nmi_unit_subject | 0.0481 ± 0.0001 | 0.0001 | 0.0000 |
| 10 | ami_unit_action | 0.0827 ± 0.0001 | -0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0480 ± 0.0001 | -0.0000 | 0.0000 |
| 100 | mi_unit_action | 0.3460 ± 0.0060 | 0.0016 | 0.0001 |
| 100 | mi_unit_subject | 0.7331 ± 0.0116 | 0.0033 | 0.0001 |
| 100 | cmi_unit_subject_given_action | 0.8811 ± 0.0120 | 0.0232 | 0.0003 |
| 100 | h_unit | 4.4339 ± 0.0158 | 4.4339 | 0.0000 |
| 100 | h_action_given_unit | 1.3657 ± 0.0060 | 1.7101 | 0.0001 |
| 100 | h_subject_given_unit | 2.0255 ± 0.0116 | 2.7553 | 0.0001 |
| 100 | nmi_unit_action | 0.1126 ± 0.0018 | 0.0005 | 0.0000 |
| 100 | nmi_unit_subject | 0.2038 ± 0.0029 | 0.0009 | 0.0000 |
| 100 | ami_unit_action | 0.1121 ± 0.0018 | 0.0000 | 0.0000 |
| 100 | ami_unit_subject | 0.2031 ± 0.0029 | -0.0000 | 0.0000 |
| 1000 | mi_unit_action | 0.6203 ± 0.0030 | 0.0166 | 0.0003 |
| 1000 | mi_unit_subject | 1.6060 ± 0.0061 | 0.0353 | 0.0003 |
| 1000 | cmi_unit_subject_given_action | 1.7662 ± 0.0057 | 0.1577 | 0.0006 |
| 1000 | h_unit | 6.6023 ± 0.0086 | 6.6023 | 0.0000 |
| 1000 | h_action_given_unit | 1.0914 ± 0.0030 | 1.6951 | 0.0003 |
| 1000 | h_subject_given_unit | 1.1526 ± 0.0061 | 2.7234 | 0.0003 |
| 1000 | nmi_unit_action | 0.1492 ± 0.0006 | 0.0040 | 0.0001 |
| 1000 | nmi_unit_subject | 0.3431 ± 0.0015 | 0.0075 | 0.0001 |
| 1000 | ami_unit_action | 0.1458 ± 0.0006 | 0.0000 | 0.0001 |
| 1000 | ami_unit_subject | 0.3381 ± 0.0016 | -0.0000 | 0.0001 |

## lara, raw, W=50

| Ku | Quantity | Observed ± seed SD | Reference mean | Mean within-cell reference SD |
| --- | --- | --- | --- | --- |
| 10 | mi_unit_action | 0.1598 ± 0.0006 | 0.0006 | 0.0001 |
| 10 | mi_unit_subject | 0.1224 ± 0.0005 | 0.0013 | 0.0001 |
| 10 | cmi_unit_subject_given_action | 0.1760 ± 0.0004 | 0.0091 | 0.0004 |
| 10 | h_unit | 2.1400 ± 0.0020 | 2.1400 | 0.0000 |
| 10 | h_action_given_unit | 1.5508 ± 0.0006 | 1.7100 | 0.0001 |
| 10 | h_subject_given_unit | 2.6362 ± 0.0005 | 2.7573 | 0.0001 |
| 10 | nmi_unit_action | 0.0830 ± 0.0002 | 0.0003 | 0.0000 |
| 10 | nmi_unit_subject | 0.0500 ± 0.0002 | 0.0005 | 0.0001 |
| 10 | ami_unit_action | 0.0827 ± 0.0002 | 0.0000 | 0.0000 |
| 10 | ami_unit_subject | 0.0495 ± 0.0002 | 0.0000 | 0.0001 |
| 100 | mi_unit_action | 0.3466 ± 0.0042 | 0.0066 | 0.0003 |
| 100 | mi_unit_subject | 0.7710 ± 0.0085 | 0.0142 | 0.0004 |
| 100 | cmi_unit_subject_given_action | 0.9353 ± 0.0065 | 0.0911 | 0.0012 |
| 100 | h_unit | 4.4421 ± 0.0101 | 4.4421 | 0.0000 |
| 100 | h_action_given_unit | 1.3640 ± 0.0042 | 1.7040 | 0.0003 |
| 100 | h_subject_given_unit | 1.9876 ± 0.0085 | 2.7444 | 0.0004 |
| 100 | nmi_unit_action | 0.1127 ± 0.0013 | 0.0022 | 0.0001 |
| 100 | nmi_unit_subject | 0.2142 ± 0.0022 | 0.0039 | 0.0001 |
| 100 | ami_unit_action | 0.1107 ± 0.0013 | -0.0000 | 0.0001 |
| 100 | ami_unit_subject | 0.2110 ± 0.0022 | -0.0000 | 0.0001 |
| 1000 | mi_unit_action | 0.6511 ± 0.0026 | 0.0707 | 0.0010 |
| 1000 | mi_unit_subject | 1.6603 ± 0.0034 | 0.1538 | 0.0015 |
| 1000 | cmi_unit_subject_given_action | 1.8488 ± 0.0013 | 0.4865 | 0.0021 |
| 1000 | h_unit | 6.6021 ± 0.0081 | 6.6021 | 0.0000 |
| 1000 | h_action_given_unit | 1.0595 ± 0.0026 | 1.6399 | 0.0010 |
| 1000 | h_subject_given_unit | 1.0983 ± 0.0034 | 2.6048 | 0.0015 |
| 1000 | nmi_unit_action | 0.1567 ± 0.0008 | 0.0170 | 0.0002 |
| 1000 | nmi_unit_subject | 0.3547 ± 0.0010 | 0.0329 | 0.0003 |
| 1000 | ami_unit_action | 0.1421 ± 0.0008 | 0.0001 | 0.0002 |
| 1000 | ami_unit_subject | 0.3328 ± 0.0010 | -0.0000 | 0.0003 |
