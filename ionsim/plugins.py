"""Models loaded from your own folders, so private models never have to live in this repository.

Point the app at a folder (Models > Load models folder…). Every file in it, or in a folder one
level down, whose name ends in `_plugin.py` is imported, and the models it lists join the built-in
ones: on Home, in File > Add model, in the Instrument tab, in setups and in the Sweeps tab.

A plugin file lists its models in MODELS:

    from ionsim.gui.common import SimTab

    class MyTab(SimTab):                      # the usual model tab: settings, Run, plots
        ...

    MODELS = {"My device": (MyTab, "One line about it for the Home page.")}

The plugin's own folder is put on the import path, so it can import the helper modules next to it
(other .py files are not imported unless the plugin does). Errors in a plugin are reported, and
never stop the app from starting.
"""
import importlib.util
import os
import sys
import traceback

SUFFIX = "_plugin.py"


def plugin_files(folder):
    """The plugin files in a folder and in its sub-folders one level down, in name order."""
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if name.endswith(SUFFIX) and os.path.isfile(p):
            out.append(p)
        elif os.path.isdir(p) and not name.startswith((".", "_")):
            out += [os.path.join(p, f) for f in sorted(os.listdir(p))
                    if f.endswith(SUFFIX) and os.path.isfile(os.path.join(p, f))]
    return out


def _module_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return f"ionsim_plugin_{abs(hash(os.path.abspath(path))) % 10 ** 8}_{base}"


def load_file(path):
    """Import one plugin file; returns its MODELS {name: (tab class, description)}."""
    here = os.path.dirname(os.path.abspath(path))
    if here not in sys.path:
        sys.path.insert(0, here)
    name = _module_name(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    models = getattr(mod, "MODELS", None)
    if not isinstance(models, dict) or not models:
        raise ValueError("it has no MODELS = {name: (tab class, description)}")
    for k, v in models.items():
        if not (isinstance(k, str) and isinstance(v, tuple) and len(v) == 2 and callable(v[0])):
            raise ValueError(f"MODELS['{k}'] should be (tab class, description)")
    return models


def load_folders(folders, taken=(), loaded=None):
    """Load every plugin in the folders. Returns (models {name: (class, description, file)},
    errors [(file, message)]). Names already in `taken` (the built-in models) are refused.
    loaded: {name: file} of the models loaded before; a plugin file that is already loaded (the same
    file reached through two folders) is skipped quietly, and a second copy of a model is skipped
    with a note saying where the first one came from."""
    loaded = dict(loaded or {})
    seen = {os.path.realpath(f) for f in loaded.values()}
    models, errors = {}, []
    for folder in folders:
        if not os.path.isdir(folder):
            errors.append((folder, "folder not found"))
            continue
        for path in plugin_files(folder):
            if os.path.realpath(path) in seen:
                continue
            seen.add(os.path.realpath(path))
            try:
                found = load_file(path)
            except BaseException as e:
                tb = traceback.format_exc(limit=-3)
                errors.append((path, f"{type(e).__name__}: {e}\n{tb}"))
                continue
            for k, (cls, desc) in found.items():
                first = models[k][2] if k in models else loaded.get(k)
                if first:
                    errors.append((path, f"a model called '{k}' is already loaded from {first}, so this copy "
                                         "was skipped. Remove one of the folders (Models > Remove models folder) "
                                         "or rename the model"))
                    continue
                if k in taken:
                    errors.append((path, f"'{k}' is the name of a built-in model; rename it"))
                    continue
                models[k] = (cls, desc, path)
    return models, errors


def unload_folder(folder, loaded):
    """The models in `loaded` ({name: file}) that came from this folder, and take the folder off
    the import path. The modules stay imported until the app restarts."""
    root = os.path.realpath(folder)
    gone = [k for k, f in loaded.items() if os.path.realpath(f).startswith(root + os.sep)]
    for d in {os.path.dirname(os.path.abspath(loaded[k])) for k in gone}:
        while d in sys.path:
            sys.path.remove(d)
    return gone
