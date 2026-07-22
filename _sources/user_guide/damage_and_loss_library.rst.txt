.. _damage_and_loss_library:

Damage and loss library
-----------------------

The NHERI-SimCenter is maintaining a comprehensive Damage and Loss Model Library (DLML) in the form of a GitHub repository.
The DLML consists of published and commonly used fragility curves, as well as loss and consequence functions.
More details can be found in the repository.

The library is distributed as the ``simcenter-dlml`` Python package, which is installed automatically together with pelicun.
The model data lives inside the installed package, so it is available immediately after installation — no separate download step is needed and no network access is required at runtime.
Reference the bundled models in your inputs with the ``PelicunDefault/<method>/<file>`` syntax, where ``<method>`` is either a pelicun method alias, for example ``PelicunDefault/FEMA P-58/fragility.csv``, or a full DLML dataset ID, for example ``PelicunDefault/seismic/building/component/FEMA P-58 2nd Edition/fragility.csv``.

To update the model library independently of pelicun, upgrade the package::

    pip install --upgrade simcenter-dlml

To work with a custom copy of the library, install your checkout in place of the released package::

    pip install -e <path-to-your-DLML-checkout>

.. button-link:: https://github.com/NHERI-SimCenter/DamageAndLossModelLibrary
    :color: primary
    :shadow:

    Visit the NHERI-SimCenter damage and loss library
