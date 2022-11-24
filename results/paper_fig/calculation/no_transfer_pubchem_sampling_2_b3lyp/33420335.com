%chk=./calculation/no_transfer_pubchem_sampling_2_b3lyp/33420335.chk
%nprocshared=7
%mem=6GB
#p b3lyp/6-31g(d) opt SCF=XQC 

Gaussian input prepared by ASE

0 1
C                 5.3897579051       -0.5387052731       -3.2953562709
O                 5.7374009363       -0.1368721855       -1.9916925762
C                 4.7720629617       -0.1575132087       -1.0369457364
C                 3.4614803771       -0.5527347471       -1.2342431724
C                 2.5499284955       -0.5106761700       -0.1769395564
C                 1.1414186236       -0.8988251943       -0.3704545326
O                 0.2755005355       -0.8115578843        0.4681605356
O                 0.8961387389       -1.3905432524       -1.6010247175
C                -0.4606116609       -1.7305931021       -1.8292104359
C                -1.3575985060       -0.5127346714       -1.5461851214
O                -1.0316024137        0.5961374604       -1.8985812842
N                -2.5155796419       -0.8573672496       -0.9245569440
C                -3.5391475697       -0.0466765411       -0.4757102753
C                -3.4792201769        1.3432511534       -0.6621127748
C                -4.5221398467        2.0961917878       -0.1947493642
C                -5.6209745921        1.5096657132        0.4490322164
C                -5.6999906958        0.1629466440        0.6447045920
C                -4.6444833818       -0.6521099278        0.1788921869
C                -4.7091679521       -2.1023417278        0.3898866142
C                -5.8912206691       -2.6546883565        1.1468055246
O                -3.8528836271       -2.8805014375       -0.0169645758
O                -6.5111669896        2.4905771228        0.8041872855
C                -5.9372513183        3.7052386700        0.3735469307
O                -4.6994143431        3.4407050213       -0.2543941671
C                 2.9528760216       -0.1030026312        1.0853193780
C                 4.2711215029        0.2760001077        1.2996094180
C                 5.1806919117        0.2746341667        0.2412682958
O                 6.4747020681        0.6917680374        0.3864970404
C                 6.6988253942        1.4114589197        1.5777713175
C                 6.0390078921        0.6926592963        2.7533714939
O                 4.6422920878        0.6428067597        2.5658481012
H                 4.6068128548        0.0990817927       -3.7171880047
H                 6.2991042111       -0.4328627035       -3.8846693641
H                 5.0606942344       -1.5823701681       -3.3198764153
H                 3.1144767785       -0.8774565231       -2.1994915418
H                -0.7445254516       -2.5824860586       -1.2029346433
H                -0.5234561738       -1.9964100357       -2.8866341024
H                -2.6733314834       -1.8431844826       -0.6789657486
H                -2.6314422630        1.7883802030       -1.1517849786
H                -6.5567629437       -0.2523262895        1.1432463680
H                -6.8169596566       -2.4430795838        0.6170787554
H                -5.9451111086       -2.2204849192        2.1419804194
H                -5.7706178723       -3.7319433536        1.2334830130
H                -6.6127421557        4.2011001944       -0.3412452354
H                -5.7755002029        4.3664700182        1.2396217600
H                 2.2502540905       -0.0815883548        1.9015373673
H                 6.2770873750        2.4221139729        1.4909307346
H                 7.7806670948        1.4735486740        1.7024310334
H                 6.2065354698        1.2228532101        3.6918727851
H                 6.4440611354       -0.3259508928        2.8262873723


