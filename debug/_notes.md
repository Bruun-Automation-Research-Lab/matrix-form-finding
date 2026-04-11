# Notes for me

1. DR with no preload is basically FD_fixed with Q = 1

2. DR_leap: turn off energy peak reset for Struct_2 to see oscillation Peaks in Energy - correspond to where L_total = 0
   in the undamped solution so start of oscillation happens here

3. If doing the validation struct_2 (Veenendaal) structure, make sure to set to identity matrix
   `self.L_0 = np.eye(self.L.shape[0])` in initialization. Not to original lengths of the network, otherwise gives wrong
   answer. EA/L = 1 is what we are trying to initilize to (what it allows)
