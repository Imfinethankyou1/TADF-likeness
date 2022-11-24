%chk=./calculation/no_transfer_pubchem_sampling_0_b3lyp/102365890.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                -4.2265231772        1.6993746887        2.8585530917
C                -4.7986505889        1.1047506603        1.5807520991
O                -3.7812803922        0.9633770115        0.5853351782
C                -2.9902191997       -0.0953918440        0.6624958612
O                -3.1358475821       -1.0058977930        1.4390012218
C                -1.8662809565        0.0176888633       -0.3609664737
C                -0.6807325342        0.5804069838        0.4384746635
O                -0.7414741191        1.7137067149        0.8729846276
N                 0.3713457833       -0.2313440201        0.6710837951
C                 1.4641992724        0.2996815590        1.4760831860
C                 2.6510514281        0.7157333368        0.6472850248
C                 2.5128743019        1.7369200257       -0.2871190374
C                 3.5862183929        2.1117489114       -1.0737392405
C                 4.8105846773        1.4798154139       -0.9274505143
C                 4.9577797773        0.4714967205        0.0091177316
C                 3.8809417209        0.0883093744        0.7904581657
C                 0.5213253610       -1.5948603205        0.1907383675
C                 1.3912178805       -1.6795322746       -1.0872511506
C                 0.5570703968       -1.9565464580       -2.3369701223
C                -0.6959351592       -1.0680371887       -2.3405930159
C                -0.2441651910        0.2996668747       -2.4815958268
N                 0.1587483062        1.3551919498       -2.6552995659
C                -1.5636095613       -1.3005871958       -1.0711488922
C                -0.8482862624       -2.2030483446       -0.0702335735
H                -5.0299782201        1.9053325288        3.5599902949
H                -3.6981993157        2.6219325129        2.6321004482
H                -3.5282250939        1.0013966736        3.3137206564
H                -5.5395085970        1.7688940884        1.1317789898
H                -5.2435386144        0.1233315608        1.7766650735
H                -2.1699544917        0.8018435539       -1.0613110211
H                 1.0642774631        1.1723765160        2.0009477890
H                 1.7560412217       -0.4583419264        2.2094923169
H                 1.5582730546        2.2275616269       -0.3992787179
H                 3.4654200779        2.8992541750       -1.8018627539
H                 5.6496218109        1.7743579980       -1.5405373850
H                 5.9116331217       -0.0211350994        0.1302866024
H                 3.9985174417       -0.7028840683        1.5181210176
H                 1.0219504859       -2.1482807522        0.9944654038
H                 1.9239662857       -0.7372572891       -1.2165399168
H                 2.1378204758       -2.4664524725       -0.9724696727
H                 1.1549875327       -1.7484437683       -3.2266312584
H                 0.2542439576       -3.0037198181       -2.3812106498
H                -1.2924101483       -1.2886106373       -3.2342385077
H                -2.5102700918       -1.7629885846       -1.3695321736
H                -0.7421466200       -3.2210843365       -0.4428468215
H                -1.4328763110       -2.2397061312        0.8488946852


