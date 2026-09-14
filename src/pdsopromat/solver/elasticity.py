"""Сборка и решение квазистатической задачи упругости с деградацией и теплом.

Эталонный путь контракта C3. Определяющие соотношения сюда не переписываются:
напряжение из деформации считает :mod:`pdsopromat.physics.elasticity`, тот же
код, которым пользуется суррогатный путь. Здесь только слабая форма, сборка,
граничные условия и решение.

Функции оставлены низкоуровневыми (отдельно матрица, отдельно правые части),
чтобы набор верификации мог собрать задачу с искусственным решением и объёмной
силой, не проходя через обвязку конкретного случая нагружения. Верифицировать
надо оператор, а не то, как он обёрнут.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from skfem import BilinearForm, LinearForm, condense, solve
from skfem.helpers import ddot, dot, sym_grad, trace

from pdsopromat.core.spec import CaseSpec, MaterialParams
from pdsopromat.physics.elasticity import plane_stress_moduli, stress_from_strain
from pdsopromat.solver.space import FemSpace

FloatArray = npt.NDArray[np.float64]


@BilinearForm
def stiffness_form(u: Any, v: Any, w: Any) -> Any:
    """m·[2μ ε(u):ε(v) + λ tr(ε(u))tr(ε(v))] — плоское напряжённое состояние."""
    strain_u = sym_grad(u)
    strain_v = sym_grad(v)
    return w["multiplier"] * (
        2.0 * w["mu"] * ddot(strain_u, strain_v)
        + w["lam"] * trace(strain_u) * trace(strain_v)
    )


@LinearForm
def _thermal_form(v: Any, w: Any) -> Any:
    """m·2(λ+μ)·α·ΔT·tr(ε(v)).

    Получается подстановкой изотропной ε_th = αΔT·I в упругий билинейный оператор:
    ε_th:ε(v) = αΔT·tr(ε(v)), а tr(ε_th) = 2αΔT.
    """
    return (
        w["multiplier"]
        * 2.0
        * (w["lam"] + w["mu"])
        * w["alpha"]
        * w["delta_t"]
        * trace(sym_grad(v))
    )


@LinearForm
def _body_force_form(v: Any, w: Any) -> Any:
    return dot(w["force"], v)


@dataclass(frozen=True)
class MechanicsSolution:
    """Результат одного решения механики.

    ``residual_norm`` возвращается всегда, в том числе для эталонного пути, где он
    заведомо мал. Это сделано намеренно: суррогат обязан отдавать ту же структуру,
    и сравнение двух путей не должно упираться в разные наборы полей.
    """

    displacement: FloatArray
    stress: FloatArray
    residual_norm: float


def assemble_stiffness(
    space: FemSpace, material: MaterialParams, multiplier: FloatArray
) -> sp.csr_matrix:
    """K(m) для узлового поля множителя жёсткости ``multiplier`` формы ``(nx, ny)``."""
    lam, mu = plane_stress_moduli(material)
    matrix: sp.csr_matrix = stiffness_form.assemble(
        space.vector,
        multiplier=space.scalar.interpolate(space.grid.as_dofs(multiplier)),
        lam=lam,
        mu=mu,
    )
    return matrix


def assemble_thermal_load(
    space: FemSpace,
    material: MaterialParams,
    multiplier: FloatArray,
    temperature: FloatArray,
) -> FloatArray:
    """Правая часть от стеснённого теплового расширения."""
    lam, mu = plane_stress_moduli(material)
    delta_t = temperature - material.reference_temperature
    load: FloatArray = _thermal_form.assemble(
        space.vector,
        multiplier=space.scalar.interpolate(space.grid.as_dofs(multiplier)),
        delta_t=space.scalar.interpolate(space.grid.as_dofs(delta_t)),
        lam=lam,
        mu=mu,
        alpha=material.thermal_expansion,
    )
    return load


def assemble_traction(space: FemSpace, edge: str, traction: tuple[float, float]) -> FloatArray:
    """Постоянная поверхностная нагрузка на одной кромке, Н/м²."""
    facet = space.facet_basis(edge)

    @LinearForm
    def form(v: Any, w: Any) -> Any:
        ones = np.ones_like(w.x[0])
        return dot(np.array([traction[0] * ones, traction[1] * ones]), v)

    load: FloatArray = form.assemble(facet)
    return load


def assemble_body_force(space: FemSpace, force: FloatArray) -> FloatArray:
    """Объёмная сила, заданная в точках квадратуры, формы ``(2, n_эл, n_кв)``.

    Нужна набору верификации: метод искусственных решений подбирает правую часть
    под выбранное точное решение, и без этого входа порядок сходимости проверить
    нечем.
    """
    load: FloatArray = _body_force_form.assemble(space.vector, force=force)
    return load


def solve_displacement(
    stiffness: sp.csr_matrix,
    load: FloatArray,
    dirichlet_dofs: FloatArray,
    dirichlet_values: FloatArray | None = None,
) -> FloatArray:
    """Решение с исключением закреплённых степеней свободы."""
    if dirichlet_values is None:
        return np.asarray(solve(*condense(stiffness, load, D=dirichlet_dofs)))

    prescribed = np.zeros(stiffness.shape[0])
    prescribed[dirichlet_dofs] = dirichlet_values
    return np.asarray(solve(*condense(stiffness, load, x=prescribed, D=dirichlet_dofs)))


def equilibrium_residual(
    stiffness: sp.csr_matrix,
    load: FloatArray,
    displacement: FloatArray,
    dirichlet_dofs: FloatArray,
) -> float:
    """‖f − K·u‖ / ‖f‖ по свободным степеням свободы.

    Один матвек, без решения системы — в этом весь смысл. Невязка на предсказанном
    суррогатом поле вычисляется дёшево и строго, тогда как «проверить, не вышли ли
    мы из распределения, запустив МКЭ» оплачивает ровно ту стоимость, которой
    суррогат должен был избежать.

    На закреплённых степенях свободы невязка смысла не имеет: там уравнение
    заменено граничным условием.
    """
    free = np.ones(stiffness.shape[0], dtype=bool)
    free[dirichlet_dofs] = False

    residual = (load - stiffness @ displacement)[free]
    reference = np.linalg.norm(load[free])
    if reference == 0.0:
        return float(np.linalg.norm(residual))
    return float(np.linalg.norm(residual) / reference)


def nodal_stress(
    space: FemSpace,
    spec: CaseSpec,
    displacement: FloatArray,
    multiplier: FloatArray,
    temperature: FloatArray,
) -> FloatArray:
    """Напряжение, спроецированное в узлы, форма ``(nx, ny, 3)``.

    Деформация в точках квадратуры разрывна между элементами; проекция в узловое
    Q1-пространство сглаживает её по методу наименьших квадратов. Это же поле идёт
    в датасет и на вход сети, поэтому восстановление обязано быть одним и тем же
    во всех путях.
    """
    strain_tensor = sym_grad(space.vector.interpolate(space.grid.as_dofs(displacement)))
    strain = np.stack(
        [strain_tensor[0][0], strain_tensor[1][1], strain_tensor[0][1]], axis=-1
    )

    delta_t = temperature - spec.material.reference_temperature
    stress = stress_from_strain(
        strain,
        spec.material,
        delta_temperature=space.scalar.interpolate(space.grid.as_dofs(delta_t)),
        multiplier=space.scalar.interpolate(space.grid.as_dofs(multiplier)),
    )

    projected = np.stack([space.scalar.project(stress[..., i]) for i in range(3)], axis=-1)
    return projected.reshape(space.grid.nx, space.grid.ny, 3)
