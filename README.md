# SFFIT

## Installation

We recommend installing *SFFIT* in a Python or conda virtual environment. Activate the environment, then clone the repository and install the package:
```
git clone https://github.com/as2875/sffit.git
cd sffit
pip install .
```
The version of JAX installed by default does not have GPU support. We highly recommend installing a version of JAX with GPU support (find the package for your architecture here: <https://docs.jax.dev/en/latest/installation.html>).

The [Monomer Library](https://github.com/MonomerLibrary/monomers) must be installed for atom typing to work. Its path should be specified in the environmental variable `$CLIBD_MON`.

## Tests

You can run the included tests using:
```
cd tests
python -m unittest
```

## Tutorial

### Adding scattering factors to an mmCIF file

Scattering factors factors for a range of atom types are available on [Zenodo](https://zenodo.org/uploads/17084047) in JSON format. Scattering factors in a JSON file can be added to an existing mmCIF file using:
```
sffit mmcif --params /path/to/params.json --models /path/to/atomic/model.cif -o /path/to/output.cif
```
You can then refine the atomic model using [Servalcat](https://github.com/keitaroyam/servalcat). Specify `--source custom` when calling Servalcat to use the scattering factors stored in the mmCIF file.

### Fitting scattering factors to data

#### Quickstart

First, fit scattering factors to some cryo-EM data:
```
sffit gp --maps /path/to/map.mrc --models /path/to/atomic/model.cif -o /path/to/params.npz -oi /path/to/intermediate.npz
```
The output is a NumPy NPZ file. The fields of this file are documented below. You can generate a JSON file from the output:
```
sffit mmcif --params /path/to/params.npz -ii /path/to/intermediate.npz -oj /path/to/output.json
```
The fields in the JSON file are documented in the [Zenodo upload](https://zenodo.org/uploads/17084047).

> [!NOTE]
> By default, *SFFIT* uses a line search to find the power likelihood weight. If you find the resulting weight gives poor performance, try changing it by setting the `--weight` option.

#### Options

| option | description |
| --- | --- |
| `--maps` | Paths to cryo-EM maps used for fitting. |
| `--models` | Paths to atomic models used for fitting, should be in the same order as maps. |
| `-oi`, `-ii` | Path to an output file that stores results of intermediate calculations. The program can be run once specifying `--maps`, `--models` and `-oi`. In subsequent runs the path given to `-oi` can be passed to `-ii` (without specifying `--maps` and `--models`) to save time. |
| `--masks` | (optional) Masks to apply to maps before fitting. It is not recommended to specify this option. |
| `--nbins` | (optional) Number of frequency bins. It is not recommended to specify this option. |
| `--rcut` | (optional) Cutoff radius (in Å) for evaluation of atomic contributions to the density. Try increasing it if the program produces unsatisfactory results. |
| `--no-change-h` | (optional) Use hydrogen atom positions specified in the atomic model. |
| `--weight` | (optional) Power likelihood weight. Determined automatically by default. |

#### Fields in output NPZ file

| field | description |
| --- | --- |
| `soln` | Scattering factors. Dimensions: number of frequency bins × number of atom types. |
| `var` | Posterior variance of scattering factors. Dimensions: number of frequency bins × number of atom types. |
| `freqs` | Centres of the frequency bins (in 1/Å). |
| `aty` | Machine-readable descriptions of each atom type. These are converted to human-readable descriptions when running `sffit mmcif`. |
| `weights` | The power likelihood weights evaluated during the line search. |
| `loss` | The score of each weight, larger is better. |
| `scale`, `beta` | Covariance hyperparameters. |
