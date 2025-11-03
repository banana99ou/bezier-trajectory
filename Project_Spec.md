베지에 곡선 기반 경로 최적화
인공위성이 ISS에 접근하는 시나리오

# equality constraints
ISS orbit altitude = 423km AMSL
chaser altitude = 300km AMSL

# nonlinear constraints subject to linearization via segmentation
KOZ = 100km AMSL

# de casteljau
segment bezier curve with de casteljau algorithm

# Linearization of KOZ
for each segment of curve j, calc CG of control points, gen unit vector Nj from center of KOZ to CG, gen supporting half space with this unit vector Nj. this is KOZ linearized to inequality constraints for this segment.

# boundary condition → position, velocity, acc boundary condition as equality constraints.

p(0) = P0, p(1) = PN
v(0) = N (P1− P0), v(1) = N (PN− PN−1)
a(0) = N (N− 1)(P2− 2P1 + P0)

# derivative of curve in control point space
for control point of bezier curve P, velocity of curve is V = EDP.
and acc of curve is A = EDV = EDEDP.
where E2→3 = {1, 0, 0; 2/3, 1/3, 0; 0, 1/3, 2/3; 0, 0, 1}.

and D3→2 = 3 * {−1, 1, 0, 0; 0, −1, 1, 0; 0, 0, −1, 1}. #D(for derivative) reduces order of P by 1 and E increases order by 1

$$
[\mathbf{D}]_{i,j} = N \times
\begin{cases}
-1, & \text{if } j = i \\
1, & \text{if } j = i + 1 \\
0, & \text{otherwise}
\end{cases}
$$

$$
[\mathbf{E}_{\,{N \to N+1}}]_{i,j} =
\begin{cases}
1, & \text{if } i = 1 \text{ and } j = 1 \\
1, & \text{if } i = N + 2 \text{ and } j = N + 1 \\
\dfrac{N + 2 - i}{N + 1}, & \text{if } 2 \le i \le N + 1 \text{ and } j = i - 1 \\
\dfrac{i - 1}{N + 1}, & \text{if } 2 \le i \le N + 1 \text{ and } j = i \\
0, & \text{otherwise}
\end{cases}
$$

# Cost function
geometric acc of bezier - gravitational acc on each point on bezier  # represents fuel usage by excluding influence of gravity from cost Fuc

# QP
use SciPy trust-constr

# output
current path figures for 6 diff seg count [2, 4, 8, 16, 32, 64]
current performance figures.
copy of performance figures but calc time versus order of curve [quadratic, cubic, 4th]
current figures of pos, vel, acc for each seg counts.