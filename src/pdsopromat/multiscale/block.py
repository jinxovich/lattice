"""Один блок нагружения: связка трёх процессов при замороженном повреждении.

Здесь сходится вся физика. Внутри блока поле повреждаемости заморожено, и порядок
такой:

1. Температура проходит цикл «подъём — выдержка — спад» с шагами по времени;
   проводимость деградирована текущим повреждением.
2. В нескольких точках цикла решается механика: температурное расширение плюс
   постоянная механическая нагрузка, жёсткость деградирована тем же полем.
3. Из напряжения считается эквивалентное по Хейхёрсту, сглаживается нелокальным
   скринингом и интегрируется по времени в движущую силу Φ.

Заморозка повреждения внутри блока законна: приращение за один блок мало. Вся
контролируемая ошибка возникает не здесь, а на прыжке через много блоков, где Φ
считается постоянной, — и там она оценивается удвоением шага.

Связь двусторонняя по существу. Температура влияет на напряжения через
расширение, напряжения — на повреждение, повреждение — обратно на жёсткость
и на теплопроводность. Односторонней цепочки, за которую карточку проекта можно
было бы упрекнуть, здесь нет.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pdsopromat.core.grid import material_mask
from pdsopromat.core.spec import CaseSpec
from pdsopromat.physics.damage import damage_driving_integral, hayhurst_equivalent_stress
from pdsopromat.physics.elasticity import degradation_multiplier
from pdsopromat.physics.loading import block_sample_times, boundary_temperature
from pdsopromat.solver.assembly import StiffnessAssembler
from pdsopromat.solver.elasticity import assemble_thermal_load, assemble_traction, nodal_stress
from pdsopromat.solver.linear import WarmSolver
from pdsopromat.solver.screening import ScreeningOperator
from pdsopromat.solver.space import FemSpace
from pdsopromat.solver.thermal import ThermalOperator

FloatArray = npt.NDArray[np.float64]
TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class BlockResult:
    """Итог одного блока — вход для прыжка и строка будущего сертификата."""

    driving_integral: FloatArray
    temperature: FloatArray
    peak_equivalent_stress: float
    compliance: float
    max_residual: float
    solves: int
    iterations: int


@dataclass(frozen=True)
class BlockEvaluator:
    """Связка на одну постановку. Тяжёлые объекты строятся один раз на прогон."""

    space: FemSpace
    spec: CaseSpec
    mask: FloatArray
    stiffness: StiffnessAssembler
    thermal: ThermalOperator
    screening: ScreeningOperator
    solver: WarmSolver
    mechanical_dofs: FloatArray
    thermal_dofs: FloatArray
    samples: int
    thermal_steps: int

    @classmethod
    def build(
        cls,
        space: FemSpace,
        spec: CaseSpec,
        *,
        samples: int = 5,
        thermal_steps_per_sample: int = 4,
    ) -> BlockEvaluator:
        mask = material_mask(space.grid, spec.geometry, spec.mesh.interface_fraction)

        # Обе осевые линии — плоскости симметрии задачи о центральном отверстии
        # под растяжением. Закрепление по ним снимает жёсткие смещения точно,
        # без точечных пиннингов, которые сами возмущают поле там, где приложены.
        half_width = 0.5 * spec.geometry.width
        half_height = 0.5 * spec.geometry.height
        mechanical = np.concatenate(
            [
                space.vector.get_dofs(
                    lambda x: np.abs(x[0] - half_width) < TOLERANCE
                ).all("u^1"),
                space.vector.get_dofs(
                    lambda x: np.abs(x[1] - half_height) < TOLERANCE
                ).all("u^2"),
            ]
        )
        boundary = np.asarray(space.scalar.get_dofs().all())

        return cls(
            space=space,
            spec=spec,
            mask=mask,
            stiffness=StiffnessAssembler.build(space, spec.material),
            thermal=ThermalOperator.build(space, spec.material),
            screening=ScreeningOperator.build(space, spec.damage.nonlocal_length),
            solver=WarmSolver(space),
            mechanical_dofs=mechanical,
            thermal_dofs=boundary,
            samples=samples,
            thermal_steps=thermal_steps_per_sample * (samples - 1),
        )

    def initial_temperature(self) -> FloatArray:
        return np.full(
            (self.space.grid.nx, self.space.grid.ny), self.spec.load.temperature_cold
        )

    def _traction(self) -> FloatArray:
        """Двустороннее растяжение вдоль x, согласованное с симметричным закреплением."""
        stress = self.spec.load.far_field_stress
        return assemble_traction(self.space, "right", (stress, 0.0)) + assemble_traction(
            self.space, "left", (-stress, 0.0)
        )

    def evaluate(
        self, damage: FloatArray, temperature: FloatArray, *, guess: FloatArray | None = None
    ) -> BlockResult:
        """Прогоняет один блок и возвращает движущую силу Φ и состояние на конце."""
        multiplier = degradation_multiplier(self.mask, damage, self.spec.damage)
        stiffness = self.stiffness.assemble(multiplier)
        traction = self._traction()

        _, weights = block_sample_times(self.spec.load, self.samples)
        sample_stride = self.thermal_steps // (self.samples - 1)
        step = self.spec.load.period / self.thermal_steps
        stepper = self.thermal.stepper(multiplier, step, self.thermal_dofs)

        equivalent = np.empty((self.samples, self.space.grid.nx, self.space.grid.ny))
        residuals: list[float] = []
        iterations = 0
        current = temperature
        displacement = guess

        for index in range(self.thermal_steps + 1):
            if index > 0:
                phase = np.asarray(index * step / self.spec.load.period)
                current = stepper.advance(
                    current, float(boundary_temperature(self.spec.load, phase))
                )
            if index % sample_stride:
                continue

            load = traction + assemble_thermal_load(
                self.space, self.spec.material, multiplier, current
            )
            displacement, report = self.solver.solve(
                stiffness, load, self.mechanical_dofs, guess=displacement
            )
            residuals.append(report.residual)
            iterations += report.iterations

            stress = nodal_stress(
                self.space,
                self.spec,
                self.space.grid.as_vector_field(displacement),
                multiplier,
                current,
            )
            # Скрининг применяется к движущей силе, а не к повреждению: так поставлена
            # неявная градиентная модель. Заодно он гасит завышенные значения у самой
            # погружённой границы, где поле восстанавливается ненадёжно.
            equivalent[index // sample_stride] = self.screening.apply(
                hayhurst_equivalent_stress(stress, self.spec.damage.triaxiality_alpha)
            )

        # Податливость меряется отдельным решением на чистой механической нагрузке
        # при однородной опорной температуре. Так метрика отказа остаётся мерой
        # жёсткости и не смешивается с текущим тепловым состоянием.
        reference, report = self.solver.solve(
            stiffness, traction, self.mechanical_dofs, guess=displacement
        )
        residuals.append(report.residual)
        iterations += report.iterations

        driving = damage_driving_integral(equivalent, weights, self.spec.damage)

        return BlockResult(
            driving_integral=driving,
            temperature=current,
            peak_equivalent_stress=float(equivalent.max()),
            compliance=float(traction @ reference),
            max_residual=max(residuals),
            solves=self.samples + 1,
            iterations=iterations,
        )
