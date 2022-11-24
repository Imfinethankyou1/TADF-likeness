%chk=calculation/no_transfer_pubchem_sampling_1_b3lyp/10708369_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_1_b3lyp/10708369_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 4.5132123116        1.2824231988       -1.4760754388
C                 4.2005393959        0.0920786877       -0.5652435085
C                 4.8012981260        0.3033671672        0.8285029181
C                 4.6646125491       -1.2197431658       -1.2071654474
O                 2.7306257405        0.1259796096       -0.4920578904
C                 2.0461298516       -0.7873319488        0.2389603745
O                 2.5379335573       -1.7008162820        0.8874257223
N                 0.6990443271       -0.5452909836        0.1487338693
C                 0.1537568906        0.5818569181       -0.6188653020
C                -1.3744413560        0.5345380729       -0.6081702634
C                -1.9752246074        0.5276836340        0.8367623692
C                -1.7353806687        1.8629369924        1.4165538759
N                -1.5119202566        2.9138806180        1.8575921010
C                -3.4859331689        0.2406311238        0.7602696636
C                -4.4487925170        1.2312232318        0.9829178659
C                -5.8103231358        0.9396624151        0.8698243020
C                -6.2309166825       -0.3446866828        0.5291298634
C                -5.2782229268       -1.3398103526        0.3014107158
C                -3.9200084882       -1.0486601511        0.4156719083
C                -1.1988393603       -0.5299520971        1.7223476100
C                -0.2213689575       -1.3844065469        0.9093693083
H                 4.0473830962        1.1515237963       -2.4583579121
H                 5.5953750211        1.3748572996       -1.6157090547
H                 4.1395208723        2.2135463879       -1.0378822491
H                 4.5634538956       -0.5320505776        1.4884208000
H                 4.4160167340        1.2280988483        1.2721929058
H                 5.8906232743        0.3931925097        0.7488735778
H                 4.4237369894       -2.0733127472       -0.5719957104
H                 5.7488963539       -1.1888230670       -1.3633557456
H                 4.1852010937       -1.3559518188       -2.1830658229
H                 0.5058999421        1.5327201360       -0.1989949098
H                 0.5138431642        0.5379593162       -1.6523530265
H                -1.7687339561        1.3922029847       -1.1617193565
H                -1.7275617643       -0.3639031284       -1.1241047045
H                -4.1390711402        2.2357820777        1.2544758672
H                -6.5405023744        1.7233882904        1.0523309394
H                -7.2900764888       -0.5707115278        0.4427800804
H                -5.5910313049       -2.3460512621        0.0358663751
H                -3.1946190110       -1.8378713142        0.2348208415
H                -0.6343045340       -0.0048355906        2.4984038616
H                -1.9192917287       -1.1755777313        2.2326004866
H                 0.3833330248       -2.0037447610        1.5728406202
H                -0.7638717831       -2.0659985799        0.2375635196


