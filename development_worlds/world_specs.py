"""Public training-catalogue metadata, never included in an agent mission.

IDs 1--6 reproduce starter v15. IDs 7--20 are synthetic training tasks, not
predictions of the competition's unknown worlds. All are second-order ODEs.
"""
from __future__ import annotations

WORLDS = tuple(str(i) for i in range(1, 21))
SCALAR_WORLDS = frozenset(('1', '2', '4', '7', '8', '9', '12'))
HIDDEN_WORLDS = frozenset(('15', '16', '20'))
TIME_WORLDS = frozenset(('6', '17', '18', '19', '20'))

# r is a two-dimensional position vector; R=|r|; s denotes softening or
# Gaussian width. Sources in 14 are visible and fixed at (-2,0), (2,0).
_ROWS = [
    ('1', '二维中心场', 1, '-G*p1/(4*pi*p2*sqrt(R^2+0.05^2)) * r/R', 'radial_inverse_distance'),
    ('2', '各向同性谐振场', 1, '-k*(p1/p2)*r', 'harmonic'),
    ('3', '旋转各向异性场', 2, '-Rot(theta) diag(kx,ky) Rot(theta)^T r', 'anisotropic_harmonic'),
    ('4', '指数屏蔽中心场', 2, '-G*p1*exp(-rho/lambda)/(2*pi*p2*rho) * r/R', 'screened_radial'),
    ('5', '交叉四次势', 3, '(-a*x-c*x^3-d*x*y^2, -b*y-c*y^3-d*y*x^2)', 'coupled_quartic'),
    ('6', '旋转主轴场', 3, '-Rot(pi*t/period+phase) diag(kx,ky) Rot(...)^T r', 'rotating_anisotropy'),
    ('7', '软化开普勒场', 1, '-G*(p1/p2)*r/(R^2+s^2)^(3/2)', 'softened_inverse_square'),
    ('8', '饱和回复场', 1, '-k*(p1/p2)*r/sqrt(1+R^2/L^2)', 'saturating_spring'),
    ('9', '径向硬化振子', 2, '-(p1/p2)*(k+c*R^2)*r', 'radial_duffing'),
    ('10', '双阱场', 2, '(a*x-b*x^3, -ky*y)', 'double_well'),
    ('11', '周期晶格场', 2, '-k*r-A*q*(sin(q*x),sin(q*y))', 'confined_periodic_lattice'),
    ('12', '高斯势阱', 2, '-(p1/p2)*A*exp(-R^2/(2*s^2))*r', 'gaussian_well'),
    ('13', '环形势阱', 2, '-k*(R^2-radius^2)*r', 'mexican_hat'),
    ('14', '可见双源场', 3, '-k*r-sum_i G_i*(r-c_i)/(|r-c_i|^2+s^2)^(3/2); c1=(-2,0),c2=(2,0)', 'visible_binary_source'),
    ('15', '偏心暗源场', 3, '-k*r-G*(r-(cx,cy))/(|r-(cx,cy)|^2+s^2)^(3/2)', 'hidden_offset_source'),
    ('16', '双暗源场', 3, '-k*r-sum_i G_i*(r-c_i)/(|r-c_i|^2+s^2)^(3/2); c1=(-d,0),c2=(d,0)', 'hidden_binary_source'),
    ('17', '呼吸谐振场', 3, '-k*(1+amplitude*sin(2*pi*t/period+phase))*r', 'pulsating_stiffness'),
    ('18', '圆周驱动场', 3, '-k*r+A*(cos(2*pi*t/period),sin(2*pi*t/period))', 'circular_drive'),
    ('19', '行波晶格场', 3, '(-k*x+A*q*sin(q*x-2*pi*t/period+phase),-k*y)', 'traveling_wave'),
    ('20', '呼吸暗源场', 3, '-k*(1+amplitude*sin(2*pi*t/period))*(r-(cx,cy))', 'pulsating_hidden_spring'),
]
WORLD_SPECS = {
    world: {'world_id': world, 'name': name, 'difficulty': difficulty,
            'formula': formula, 'mechanism': mechanism,
            'origin': 'starter_v15' if int(world) <= 6 else 'synthetic_training',
            'claims_truth': {'dynamics_order': 2, 'hidden_matter': world in HIDDEN_WORLDS,
                             'time_dependence': world in TIME_WORLDS}}
    for world, name, difficulty, formula, mechanism in _ROWS
}
