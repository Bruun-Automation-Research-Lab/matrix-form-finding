# Notes for me

1. DR with no preload is basically FD_fixed with Q = 1

2. DR_leap: turn off energy peak reset for Struct_2 to see oscillation Peaks in Energy - correspond to where L_total = 0
   in the undamped solution so start of oscillation happens here

## DR vs. PS

seems same to me, except PS has an extra term... F_0 + EA \* epsilson = F_0 + EA \* (L - L_0)/L_0

F_0 + (L - L_0) \* k_s + VelocityTerm = F_0 + (L - L_0) \* EA/L_0 + VelocityTerm
