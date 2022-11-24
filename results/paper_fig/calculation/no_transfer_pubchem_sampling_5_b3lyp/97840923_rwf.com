%chk=calculation/no_transfer_pubchem_sampling_5_b3lyp/97840923_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_5_b3lyp/97840923_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 7.7281342317        0.1210079418        0.2145023248
C                 6.3693087426       -0.4692332685        0.3515892195
C                 5.8730775607       -1.5478480180        1.0320343022
C                 4.4743077072       -1.5932427215        0.7619447891
C                 4.2048686859       -0.5346134677       -0.0695003391
C                 2.9947046542       -0.0468565309       -0.6876569116
N                 2.9201560916        0.9962331915       -1.4740137336
C                 1.6302582236        1.1990979981       -1.8883265421
C                 0.6755953835        0.3224519712       -1.4464037635
C                -0.7979263239        0.3411056844       -1.7778083767
N                -1.7124391405        0.4635665296       -0.6513759048
C                -1.6741114661        1.7233950138        0.0955164755
C                -2.8384102921        2.6637941499       -0.2262821912
C                -4.1901375468        1.9310318356       -0.2070613661
C                -4.3170227392        0.8678181697        0.8901635881
O                -3.8987392361        1.4777655403        2.1262912747
C                -3.6130983407       -0.4618762087        0.6122737762
C                -4.2557710239       -1.6045589164        1.1094700491
C                -3.7950274726       -2.8982020327        0.8772200640
C                -2.6477901753       -3.0592913906        0.1034354358
C                -1.9719576666       -1.9490650611       -0.3920659658
C                -2.4165706266       -0.6307856976       -0.1453911607
S                 1.4451775152       -0.8583262604       -0.4072570916
O                 5.3614564477        0.1549105179       -0.3225986226
H                 8.0556753490        0.1298739903       -0.8322654659
H                 7.7511715342        1.1571849175        0.5734917474
H                 8.4479942993       -0.4611960928        0.7959256320
H                 6.4414841907       -2.2275349369        1.6516321929
H                 3.7597335165       -2.3148121478        1.1338951072
H                 1.4187096867        2.0362630961       -2.5467229064
H                -0.9491312372        1.2025472822       -2.4439625716
H                -1.0822819581       -0.5411382823       -2.3592217634
H                -1.6934597520        1.4952431511        1.1644730502
H                -0.7139286503        2.2086824507       -0.1121469243
H                -2.6985531728        3.1355819711       -1.2080714932
H                -2.8352935330        3.4685218681        0.5180701593
H                -4.9968531494        2.6618759063       -0.0765966503
H                -4.3654978246        1.4353805278       -1.1691019756
H                -5.3862320373        0.6192944705        0.9749530046
H                -3.8679493192        0.7730563659        2.7926514705
H                -5.1790652235       -1.4660113962        1.6705734601
H                -4.3303044434       -3.7561887854        1.2726351164
H                -2.2607313506       -4.0526080295       -0.1092518735
H                -1.0635321187       -2.1122962962       -0.9614666459


