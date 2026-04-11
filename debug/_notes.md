# Notes for me

1. DR with no preload is basically FD_fixed with Q = 1

2. DR_leap: turn off energy peak reset for Struct_2 to see oscillation Peaks in Energy - correspond to where L_total = 0
   in the undamped solution so start of oscillation happens here

3. For struct_2 (Veenendaal) structure, for DR, setting to identity matrux `self.L_0 = np.eye(self.L.shape[0])` in
   initialization. Not to original lengths of the network, otherwise gives wrong answer.
