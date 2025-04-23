import ilthermopy as ilt
import pandas as pd

import ilthermoml as iltml
from ilthermoml.featurization import RDKitMoleculeFeaturizer, SaltFeaturizer


# Create and populate dataset. The same as in dataset.py example.
class Dataset(iltml.Dataset):
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

        return list(search["id"])

    @staticmethod
    def prepare_entry(entry: iltml.Entry) -> None:
        data = entry.data.copy()

        try:
            data = pd.DataFrame(
                {
                    "t_k": data["Temperature, K"],
                    "p_kpa": data.get("Pressure, kPa", 101.325),
                    "eta_mpa_s": data["Viscosity, Pa&#8226;s => Liquid"] * 1.0e3,
                },
            )
        except KeyError as e:
            msg = f"required data column {e} is missing"

            raise iltml.EntryError(msg) from e

        entry.data = data


dataset = Dataset()
dataset.populate()

tmp_molecule_in_entry = {}

for entry in dataset.entries:
    tmp_molecule_in_entry[entry.id] = dataset.ionic_liquids.index(entry.ionic_liquid)

molecule_in_entry = pd.DataFrame.from_dict(
    tmp_molecule_in_entry, orient="index", columns=["ionic_liquid_id"]
)
molecule_in_entry.index = molecule_in_entry.index.rename("entry_id")

data = dataset.data
data = (
    data.join(molecule_in_entry)
    .reset_index()
    .set_index(["ionic_liquid_id", "entry_id", "data_point_id"])
    .query("t_k > 297.5 & t_k < 298.5")
)

tmp_temp_visc = {}

for il in set(data.index.get_level_values(0)):
    visc = data.loc[il]["eta_mpa_s"].mean()
    temp = data.loc[il]["t_k"].mean()
    tmp_temp_visc[il] = [temp, visc]

temp_visc = pd.DataFrame.from_dict(
    tmp_temp_visc, orient="index", columns=["t_k", "eta_mpa_s"]
)
temp_visc.index = temp_visc.index.rename("ionic_liquid_id")

featurize = SaltFeaturizer(lambda x, y: (x + y) / 2, RDKitMoleculeFeaturizer())

features = pd.DataFrame.from_dict(
    {
        x: featurize(dataset.ionic_liquids[x])
        for x in set(data.index.get_level_values(0))
    },
    orient="index",
)
features.index = features.index.rename("ionic_liquid_id")

design_table = temp_visc.join(features)
design_table.to_csv("rdkit_datasest.csv")
