# Notes

## 1. Relationship between `DR` and `FD`

With no preload, the Dynamic Relaxation (`DR`) method is essentially equivalent to `FD_fixed` with

$$
Q = 1
$$

for all members.

---

## 2. Oscillation behavior in `DR_leap`

For Veenendaal validation structure, turn off the energy peak reset in `DR_leap` if you want to observe the oscillation
behavior more clearly.

In that case:

- peaks in the energy curve become visible
- these correspond to points where the total length measure reaches its oscillatory turning point
- in the undamped solution, this marks the onset of the oscillation cycle

<!-- ---

## 3. Initialization for Veenendaal validation structure

For the Veenendaal validation structure, make sure the unstressed length matrix is initialized as

    self.L_0 = np.eye(self.L.shape[0])

and **not** using the original member lengths of the starting network.

Using the original lengths gives the wrong result.

The reason is that the intended initialization is based on

$$
\frac{EA}{L_0} = 1
$$

which, in this setup, is enforced by taking all unstressed lengths equal to `1`. -->

---

## 4. Veenendaal Validation 1

Validation 1 uses the parameters:

$$
Q = 1, \qquad EA = 1, \qquad F_0 = 1
$$

and all unstressed lengths are equal.

### Recommended runs

- Run `FD_fixed` for this case
- For `SM` and `DR`, set

  self.fd_mode = "constant_q"

This ONLY works when we want to get q = 1, do this by setting preload = 1 in input

### Derivation of `L_0`

We use

$$
L_0 = \frac{EA \cdot L}{Q \cdot L + EA - F_0}
$$

Substituting the validation parameters:

$$
L_0 = \frac{(1)(1)\cdot L}{(1)\cdot L + (1)(1) - (1)}
$$

which simplifies to

$$
L_0 = \frac{L}{L} = 1
$$

So for this validation case, all unstressed lengths should be initialized as

$$
L_0 = 1
$$

---

## 5. Veenendaal Validation 2

Validation 2 uses the parameters:

$$
F = 1, \qquad EA = 0
$$

### Recommended runs

- Run `FD_iter` for this case
- For `SM` and `DR`, set

  self.fd_mode = "constant_f"

This currently does **not** work properly for `DR_leap`.

### Constant-force interpretation

Starting from

$$
F = \frac{EA}{L_0}(L - L_0) + F_0
$$

if

$$
EA = 0
$$

then this reduces to

$$
F = F_0 = 1
$$

So this case represents a constant-force formulation. Where the constant force that we want to reach in the members is
set as the preload in the input file

### Important numerical note

Although setting

$$
EA = 0
$$

works for `DR`, it does **not** work for `SM`, because the stiffness contribution disappears and the system becomes
singular. But it can also destablize DR since no elastic stiffness now, so best to do it this way

So for the stiffness-based implementation, the `EA` terms must remain nonzero numerically.

If the goal is still to enforce

$$
F = F_0 = 1
$$

then an equivalent way is to set

$$
L = L_0
$$

so that

$$
L - L_0 = 0
$$

and therefore

$$
F = \frac{EA}{L_0}(L - L_0) + F_0
= \frac{EA}{L_0}(0) + F_0
= F_0
= 1
$$

This gives the desired constant-force behavior by setting the preload to the desired force without requiring the
stiffness term to vanish
