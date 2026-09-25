"""解析対象・不純物のプリセット。新しい物質は MATERIAL_PRESETS に足す。"""

from __future__ import annotations

from dataclasses import dataclass, field

from pymatgen.core import Lattice, Structure


@dataclass(frozen=True)
class MaterialPreset:
    id: str
    name: str
    formula: str
    crystal_system: str
    space_group: str
    space_group_number: int
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    notes: str = ""
    synthesis_routes: dict[str, list[str]] = field(default_factory=dict)

    def lattice_params(self) -> dict[str, float]:
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
        }


def _nb2alc() -> Structure:
    return Structure.from_spacegroup(
        194,
        Lattice.hexagonal(3.107, 13.888),
        ["Nb", "Al", "C"],
        [[1.0 / 3.0, 2.0 / 3.0, 0.088], [1.0 / 3.0, 2.0 / 3.0, 0.75], [0.0, 0.0, 0.0]],
    )


def _lco() -> Structure:
    return Structure.from_spacegroup(
        166,
        Lattice.hexagonal(2.816, 14.054),
        ["Li", "Co", "O"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.26]],
    )


def prototype_structure(preset_id: str) -> Structure | None:
    builders = {"Nb2AlC": _nb2alc, "LCO": _lco}
    if preset_id not in builders:
        return None
    return builders[preset_id]()


def dummy_structure(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> Structure:
    """CIFも内蔵構造もないときの位置のみモデル（強度は参考値）。"""
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    return Structure(lattice, ["X"], [[0.0, 0.0, 0.0]])


def scale_structure(
    proto: Structure,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Structure:
    scaled = proto.copy()
    scaled.lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    return scaled


def structure_from_params(
    preset_id: str,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Structure:
    proto = prototype_structure(preset_id)
    if proto is None:
        return dummy_structure(a, b, c, alpha, beta, gamma)
    return scale_structure(proto, a, b, c, alpha, beta, gamma)


def crystal_system_defaults(system: str) -> dict[str, float]:
    system = system.lower()
    if system == "cubic":
        return {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system in {"hexagonal", "trigonal"}:
        return {"a": 3.0, "b": 3.0, "c": 14.0, "alpha": 90.0, "beta": 90.0, "gamma": 120.0}
    if system == "tetragonal":
        return {"a": 4.0, "b": 4.0, "c": 6.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "orthorhombic":
        return {"a": 5.0, "b": 6.0, "c": 7.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    if system == "monoclinic":
        return {"a": 5.0, "b": 6.0, "c": 7.0, "alpha": 90.0, "beta": 100.0, "gamma": 90.0}
    return {"a": 5.0, "b": 5.0, "c": 5.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}


def constrain_by_system(
    system: str, a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> tuple[float, float, float, float, float, float]:
    system = system.lower()
    if system == "cubic":
        return a, a, a, 90.0, 90.0, 90.0
    if system in {"hexagonal", "trigonal"}:
        return a, a, c, 90.0, 90.0, 120.0
    if system == "tetragonal":
        return a, a, c, 90.0, 90.0, 90.0
    if system == "orthorhombic":
        return a, b, c, 90.0, 90.0, 90.0
    if system == "monoclinic":
        return a, b, c, 90.0, beta, 90.0
    return a, b, c, alpha, beta, gamma


def fit_axes(system: str) -> list[str]:
    system = system.lower()
    if system == "cubic":
        return ["a"]
    if system in {"hexagonal", "trigonal", "tetragonal"}:
        return ["a", "c"]
    if system == "orthorhombic":
        return ["a", "b", "c"]
    if system == "monoclinic":
        return ["a", "b", "c", "beta"]
    return ["a", "b", "c", "alpha", "beta", "gamma"]


def impurity_structure(name: str) -> Structure:
    builders = {
        "NbC": lambda: Structure.from_spacegroup(
            225, Lattice.cubic(4.470), ["Nb", "C"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "Al2O3": lambda: Structure.from_spacegroup(
            167,
            Lattice.hexagonal(4.759, 12.991),
            ["Al", "O"],
            [[0.0, 0.0, 0.3523], [0.3064, 0.0, 0.25]],
        ),
        "Nb": lambda: Structure.from_spacegroup(229, Lattice.cubic(3.300), ["Nb"], [[0, 0, 0]]),
        "Al": lambda: Structure.from_spacegroup(225, Lattice.cubic(4.050), ["Al"], [[0, 0, 0]]),
        "C": lambda: Structure.from_spacegroup(227, Lattice.cubic(3.567), ["C"], [[0, 0, 0]]),
        "NaCl": lambda: Structure.from_spacegroup(
            225, Lattice.cubic(5.640), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "NbAl3": lambda: Structure.from_spacegroup(
            139,
            Lattice.tetragonal(3.845, 8.601),
            ["Nb", "Al", "Al"],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.25]],
        ),
        "Co3O4": lambda: Structure.from_spacegroup(
            227,
            Lattice.cubic(8.084),
            ["Co", "Co", "O"],
            [[0.125, 0.125, 0.125], [0.5, 0.5, 0.5], [0.26, 0.26, 0.26]],
        ),
        "CoO": lambda: Structure.from_spacegroup(
            225, Lattice.cubic(4.261), ["Co", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "Li2O": lambda: Structure.from_spacegroup(
            225, Lattice.cubic(4.619), ["O", "Li"], [[0, 0, 0], [0.25, 0.25, 0.25]]
        ),
        "Co": lambda: Structure.from_spacegroup(225, Lattice.cubic(3.545), ["Co"], [[0, 0, 0]]),
        "Li2CO3": lambda: Structure.from_spacegroup(
            15,
            Lattice.monoclinic(8.359, 4.977, 6.194, 114.83),
            ["Li", "C", "O", "O"],
            [[0.197, 0.44, 0.83], [0.0, 0.066, 0.25], [0.15, 0.88, 0.31], [0.0, 0.32, 0.25]],
        ),
    }
    if name not in builders:
        raise KeyError(f"未登録の不純物: {name}")
    return builders[name]()


KNOWN_IMPURITIES = (
    "NbC",
    "Al2O3",
    "Nb",
    "Al",
    "C",
    "NaCl",
    "NbAl3",
    "Co3O4",
    "CoO",
    "Li2O",
    "Co",
    "Li2CO3",
)

MATERIAL_PRESETS: dict[str, MaterialPreset] = {
    "Nb2AlC": MaterialPreset(
        id="Nb2AlC",
        name="Nb2AlC",
        formula="Nb2AlC",
        crystal_system="hexagonal",
        space_group="P63/mmc",
        space_group_number=194,
        a=3.107,
        b=3.107,
        c=13.888,
        gamma=120.0,
        notes="211 MAX相",
        synthesis_routes={
            "HFエッチング法": ["NbC", "Al2O3", "Nb", "Al", "C"],
            "溶融塩法": ["NbC", "Al2O3", "Nb", "Al", "NaCl"],
            "反応焼結法": ["NbC", "Al2O3", "Nb", "Al", "NbAl3"],
        },
    ),
    "LCO": MaterialPreset(
        id="LCO",
        name="LCO (LiCoO2)",
        formula="LiCoO2",
        crystal_system="trigonal",
        space_group="R-3m",
        space_group_number=166,
        a=2.816,
        b=2.816,
        c=14.054,
        gamma=120.0,
        notes="層状酸化物正極",
        synthesis_routes={
            "固相反応法": ["Co3O4", "CoO", "Li2CO3", "Li2O", "Co"],
            "ゾルゲル法": ["Co3O4", "Li2CO3", "Li2O"],
            "水熱法": ["CoO", "Co3O4", "Li2O"],
        },
    ),
}

CUSTOM_PRESET_ID = "CUSTOM"
CRYSTAL_SYSTEMS = (
    "cubic",
    "hexagonal",
    "trigonal",
    "tetragonal",
    "orthorhombic",
    "monoclinic",
    "triclinic",
)
