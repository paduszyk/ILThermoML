import ilthermopy as ilt
import pandas as pd

import ilthermoml


class Dataset(ilthermoml.Dataset):
    @staticmethod
    def get_entry_ids() -> list[str]:
        search = ilt.Search(prop="Viscosity", n_compounds=1)

        def is_binary_smiles(smiles: str) -> bool:
            return len(smiles.split(".")) == 2  # noqa: PLR2004

        search = search[
            (search["num_phases"] == 1)
            & (search["phases"] == "Liquid")
            & search["cmp1_smiles"].apply(is_binary_smiles)
        ]

        return list(search["id"])[:100]

    @staticmethod
    def prepare_entry(entry: ilthermoml.Entry) -> None:
        data = entry.data.copy()

        try:
            data = pd.DataFrame(
                {
                    "t_k": data["Temperature, K"],
                    "p_kpa": data.get("Pressure, kPa", 101.325),
                    "eta_mpa_s": data["Viscosity, Pa&#8226;s => Liquid"] * 1.0e003,
                },
            )
        except KeyError as e:
            msg = f"required column {e} is missing"

            raise ilthermoml.EntryError(msg) from e

        entry.data = data


dataset = Dataset()

dataset.populate()
