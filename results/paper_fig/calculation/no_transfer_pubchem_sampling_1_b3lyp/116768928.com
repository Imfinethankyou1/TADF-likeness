%chk=./calculation/no_transfer_pubchem_sampling_1_b3lyp/116768928.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -2.6081301676        0.5895035186        1.5410846922
C                -2.6620282874       -0.5690734987        0.9024621563
C                -3.3227293251       -1.7429838452        1.5866493652
C                -2.2848588692       -2.6399520053        2.2622762308
C                -2.2129134036       -0.8702669296       -0.5092919463
C                -0.8312226285       -0.5069174781       -1.1086547585
N                -0.8354260997        0.7638585368       -1.8160966188
C                -1.2917702227        2.0062869935       -1.2387576058
C                -1.2971117821        3.0864903012       -2.3214411343
C                -1.8776835016        4.3966029696       -1.8054603871
C                 0.4110805813       -0.8372210581       -0.2038385409
O                 0.2649792098       -2.2000363950        0.2436622244
C                 0.7899358675       -3.3107490942       -0.4413039834
C                 0.1636907711       -3.6564149971       -1.7926445950
C                 1.7170029048       -0.5973167994       -0.9892676837
C                 2.3411505355        0.7631991838       -0.6876794519
C                 2.8234034622        0.8525957356        0.7724112082
C                 4.2594142906        0.3349088484        0.8892312756
C                 2.7907037801        2.3148098134        1.2254989869
C                 1.8936424070        0.0031112168        1.6581281269
C                 0.4727279061       -0.0351590531        1.1063550128
H                -3.0180372007        0.6849565847        2.5338529605
H                -2.1548522001        1.4773633984        1.1470276296
H                -4.0284471507       -1.3835650572        2.3384687026
H                -3.8850244690       -2.3285753726        0.8552610374
H                -1.4969818094       -2.9004685109        1.5614628039
H                -1.8305088446       -2.1182532562        3.1017758295
H                -2.7501958483       -3.5508068834        2.6334915402
H                -2.2948597168       -1.9522493607       -0.6216962388
H                -2.9467251031       -0.4270564805       -1.1912244869
H                -0.7286079337       -1.2210728574       -1.9399349319
H                 0.0607848809        0.9018185394       -2.2664982155
H                -0.6788076076        2.3699384170       -0.3937617859
H                -2.3135614036        1.8578818189       -0.8853585327
H                -0.2745382666        3.2525381277       -2.6708183766
H                -1.8826003536        2.7264779304       -3.1685752154
H                -1.8681400925        5.1526541843       -2.5868732420
H                -1.2968986814        4.7673693701       -0.9630312361
H                -2.9045479667        4.2548347391       -1.4761030797
H                 0.5961017127       -4.1345110885        0.2549768766
H                 1.8799602798       -3.2380683431       -0.5633785362
H                 0.5143420365       -4.6395820363       -2.0983337197
H                 0.4475570986       -2.9405927068       -2.5597689106
H                -0.9202372316       -3.6839669481       -1.7152225967
H                 1.5400110974       -0.7028174293       -2.0632547773
H                 2.4464375760       -1.3566669092       -0.7083586917
H                 1.6171237788        1.5555947697       -0.8783038247
H                 3.1855203045        0.9271905281       -1.3617971488
H                 4.5684200203        0.3114917088        1.9321138380
H                 4.3436890501       -0.6726733441        0.4877692666
H                 4.9458290026        0.9792671013        0.3430781825
H                 3.2193897332        2.4162325592        2.2202469236
H                 3.3627412192        2.9388300202        0.5413039260
H                 1.7690946021        2.6873126754        1.2557798904
H                 2.2647981450       -1.0210376900        1.7221124019
H                 1.8823523368        0.4098194872        2.6709315762
H                 0.1132070735        0.9770613429        0.9480044388
H                -0.1876454964       -0.5219469923        1.8206101493


