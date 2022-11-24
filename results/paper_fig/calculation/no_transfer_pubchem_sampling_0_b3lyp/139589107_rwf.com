%chk=calculation/no_transfer_pubchem_sampling_0_b3lyp/139589107_rwf.chk
%rwf=calculation/no_transfer_pubchem_sampling_0_b3lyp/139589107_rwf.rwf
%nprocshared=14
%mem=3GB
#p b3lyp/6-31g(d) 6D 10F nosymm GFInput td=(50-50,nstate=2) SCF=XQC 

Gaussian input prepared by ASE

0 1
C                13.0178593658        2.2090852386        2.7471722800
C                11.7996171192        3.0251315467        2.4148246677
C                10.5283953972        2.7926313550        2.7659134807
C                 9.9928296111        1.6422912808        3.5792568560
C                 9.1431392769        0.6378629525        2.7675315719
C                 7.8592255589        1.2259429513        2.1674886169
C                 7.0040453302        0.1824368715        1.4358550581
C                 5.7147139740        0.7729297431        0.8470500565
C                 4.7925902017       -0.2491503823        0.1609648036
C                 5.3436335142       -0.8265143515       -1.1434206457
C                 4.5427010414       -1.9921771821       -1.7457365300
C                 3.0113520026       -1.8832879111       -1.7878758104
C                 2.3201469388       -3.0220217446       -2.5536769867
C                 2.6596795182       -4.4462276468       -2.0712449276
C                 2.3857200093       -4.7606086384       -0.5885386021
C                 0.9253931624       -4.6303259356       -0.1227666242
C                -0.0537218305       -5.5961806769       -0.8109594742
C                -1.4310977845       -5.6815242347       -0.1298175036
C                -2.2189260430       -4.3652010816       -0.1216258752
C                -3.5774756241       -4.4558025921        0.5874833938
C                -4.2912734510       -3.1002306856        0.6481124337
C                -5.5933091857       -3.0976659054        1.4638581544
C                -6.2224196578       -1.6920658335        1.5797278424
C                -7.2986501578       -1.3433659685        0.5463816918
O                -6.8279962397       -1.8315211522       -0.7714423849
S                -7.9051689934       -2.4330117940       -1.8111114360
O                -7.1615368428       -2.6576190302       -3.0348345150
O                -9.1440856999       -1.6861989221       -1.7503773790
O                -8.2495133114       -3.8838303368       -1.1384122938
C                -7.5378442008        0.1617527055        0.3787938749
C                -7.8312427018        0.9388212205        1.6593396450
O                -9.1616141345        0.4618450701        2.1075034106
S                -9.4457162931        0.2298487700        3.6779162728
O                -8.2264214361       -0.1711971956        4.3530237864
O               -10.6778979391       -0.5290600704        3.7358657928
O                -9.7516086803        1.7439616068        4.2141058849
C                -7.8647343467        2.4577413876        1.4656203299
C                -6.4726264479        3.0348771933        1.1607712993
C                -6.4386251906        4.5678626060        1.0787741289
C                -5.0467870901        5.1249355299        0.7248521856
C                -4.6145452275        4.8506223963       -0.7244272866
C                -3.1999981528        5.3392831211       -1.0827198608
C                -2.0367993307        4.6446328624       -0.3495349112
C                -1.9984877526        3.1150378951       -0.5078117541
C                -0.6811671604        2.4769543692       -0.0221269279
C                 0.4905722776        2.5983496323       -1.0149756796
C                 0.3909091003        1.5963439162       -2.1885971428
C                 1.3478093395        1.9243857892       -3.2942474565
C                 2.3662411870        1.2222647562       -3.8204217241
C                 2.8793949114       -0.1179599208       -3.4740757099
O                 3.6251650561       -0.7304801552       -4.2411783392
O                 2.5069902739       -0.5901327072       -2.2808011993
O                 5.4094931101        0.3030743791       -2.0831944927
S                 6.6559320895        0.3683679486       -3.1456660084
O                 7.4939932218        1.5021788250       -2.8156855560
O                 7.2226664445       -0.9686277782       -3.2754498772
O                 5.8333537620        0.7750630954       -4.4578610946
H                12.7915556133        1.3325141604        3.3604294285
H                13.5108169709        1.8570210640        1.8307504555
H                13.7591244744        2.8146245749        3.2864424423
H                11.9974272666        3.9118364933        1.8110432157
H                 9.7808491345        3.5072617233        2.4210386658
H                 9.3718040259        2.0413654572        4.3961193085
H                10.8131088680        1.0967838715        4.0586869846
H                 8.8768726042       -0.2011992728        3.4268152373
H                 9.7615407979        0.2175074247        1.9627307210
H                 7.2613986363        1.6870123681        2.9686976676
H                 8.1113934888        2.0329392192        1.4670978519
H                 7.6108087603       -0.2730125689        0.6407660664
H                 6.7483254264       -0.6316041929        2.1316171423
H                 5.1466774552        1.2550506908        1.6544849837
H                 5.9648678420        1.5645448671        0.1300080323
H                 3.8225688228        0.2159846152       -0.0502907125
H                 4.6009024449       -1.0909969148        0.8425337879
H                 6.3639379401       -1.1869894754       -0.9936492101
H                 4.7595889668       -2.8668497525       -1.1182905381
H                 4.9451530297       -2.2150083242       -2.7353049482
H                 2.6358597871       -1.8608604477       -0.7613343655
H                 2.5907747720       -2.9398784282       -3.6108916176
H                 1.2380224431       -2.8515362431       -2.4876103951
H                 2.0956139536       -5.1466722682       -2.6998611394
H                 3.7167613113       -4.6541445926       -2.2800819310
H                 2.7245299968       -5.7878839361       -0.3943913916
H                 3.0137480663       -4.1200352857        0.0470928650
H                 0.8978326145       -4.8161107930        0.9607855675
H                 0.5846331221       -3.5948843524       -0.2563488340
H                -0.1877591366       -5.3095824188       -1.8636506130
H                 0.3920480659       -6.6012032560       -0.8239109466
H                -2.0289960599       -6.4557138808       -0.6310503615
H                -1.2951587092       -6.0256569376        0.9064600645
H                -1.6244464477       -3.5843166552        0.3722475794
H                -2.3692824181       -4.0233201162       -1.1567185987
H                -3.4277851922       -4.8362700273        1.6089520180
H                -4.2140965448       -5.1949625787        0.0783427666
H                -3.6054867471       -2.3664567890        1.0983275729
H                -4.4922849789       -2.7423537776       -0.3692136465
H                -6.3184538534       -3.8009679335        1.0334162988
H                -5.3755751005       -3.4677149390        2.4739095823
H                -5.4223851867       -0.9401782777        1.5319037227
H                -6.6952262260       -1.5744516415        2.5616899810
H                -8.2286389927       -1.8678584145        0.7805206523
H                -7.5564599506       -4.5106898039       -1.4207086533
H                -6.6298395173        0.5794003815       -0.0695957804
H                -8.3556260502        0.3198693403       -0.3317144243
H                -7.1169314925        0.6849163883        2.4458613683
H               -10.6926868516        1.9338471426        4.0352560435
H                -8.5668999337        2.7054633797        0.6587482025
H                -8.2554420815        2.9116605288        2.3836150948
H                -5.7674383128        2.7071384890        1.9394095220
H                -6.1029443315        2.6179503003        0.2170619656
H                -6.7616749938        4.9835791662        2.0429040900
H                -7.1708623752        4.9127067970        0.3340569053
H                -5.0414826218        6.2106406419        0.8937408343
H                -4.3106272641        4.7064780219        1.4259167337
H                -4.6857425125        3.7774967465       -0.9426625630
H                -5.3320073961        5.3390674348       -1.3990004334
H                -3.1355570074        6.4212758470       -0.8995696348
H                -3.0562489690        5.2065838707       -2.1645205365
H                -2.0640702744        4.8882063555        0.7217503164
H                -1.0990772915        5.0757665668       -0.7262539208
H                -2.1734087502        2.8506872164       -1.5613939017
H                -2.8336598852        2.6727260996        0.0502651753
H                -0.3945016457        2.9411472122        0.9314333007
H                -0.8456839943        1.4129613587        0.1976563582
H                 1.4411228704        2.4274077304       -0.4950242212
H                 0.5401706982        3.6236868694       -1.4068999309
H                 0.5376187176        0.5788686807       -1.8222368314
H                -0.6228876540        1.6496309558       -2.6134541805
H                 1.1699375918        2.8973883652       -3.7561896856
H                 2.9050890337        1.6452107499       -4.6633876379
H                 5.0725722488        0.1386319475       -4.5802564971


