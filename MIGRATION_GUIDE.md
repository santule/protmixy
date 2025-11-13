# Migration Guide

This document explains the restructuring changes made to prepare the codebase for academic publishing.

## File Renaming

| Old Name | New Name | Location |
|----------|----------|----------|
| `settings.py` | `settings.py` | `config/` |
| `model_loader.py` | `model_loader.py` | `src/utils/` |
| `msat_evolver_manifold.py` | `protein_evolver.py` | `src/` |
| `msat_evolver_helpers_manifold.py` | `evolution_utils.py` | `src/` |

## Directory Structure

### Before
```
protmixy/
├── settings.py
├── model_loader.py
├── msat_evolver_manifold.py
└── msat_evolver_helpers_manifold.py
```

### After
```
protmixy/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── MIGRATION_GUIDE.md
├── config/
│   ├── __init__.py
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── protein_evolver.py
│   ├── evolution_utils.py
│   └── utils/
│       ├── model_loader.py
│       ├── evaluator.py
│       ├── helpers.py
│       └── msa_output.py
├── scripts/
│   └── run_evolution.py
├── docs/
│   └── methods.md
└── data/
    └── .gitkeep
```

## Import Changes

### Old Imports
```python
from model_loader import ModelLoader
from config.settings import DEVICE
import msat_evolver_helpers_manifold as helpers
```

### New Imports
```python
from src.utils.model_loader import ModelLoader
from config.settings import DEVICE
from src.evolution_utils import *
```

## Running the Code

### Old Way
```python
# Direct execution of msat_evolver_manifold.py
python msat_evolver_manifold.py
```

### New Way
```bash
# Using the main script
python scripts/run_evolution.py --start dummy1 --end dummy2 --seed 42
```

## Configuration Changes

The `config/settings.py` file has been enhanced with:
- Better documentation and comments
- Organized sections (Experiment, Paths, Models, Hyperparameters, Visualization)
- A `print_config()` function for displaying settings
- Renamed some variables for clarity (e.g., `esm2_*` → `ESM2_*`)

## Missing Dependencies

The following modules are referenced but not included in the repository:
- `common_utils.helpers` - You'll need to add this or update imports
- `generator.utils.evaluator_manifold` - You'll need to add this or update imports

### Action Required

If these modules exist elsewhere:
1. Copy them to appropriate locations in the new structure
2. Update their imports to match the new structure
3. Add them to git

If they don't exist:
1. Remove or comment out references to them
2. Implement the required functionality

## Git Repository

The repository has been initialized with:
- `.gitignore` configured for Python projects
- Initial commit with all restructured files
- Proper exclusion of data files, models, and outputs

## Next Steps

1. **Add missing dependencies**: Copy or implement `helpers` and `evaluator_manifold` modules
2. **Update data paths**: Ensure `ROOT_PATH` in `config/settings.py` points to correct location
3. **Test the code**: Run a simple test to ensure imports work correctly
4. **Add your data**: Place MSA files in `data/` directory
5. **Configure remote**: Add GitHub/GitLab remote and push

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

## Backward Compatibility

The old files (`settings.py`, `model_loader.py`, etc.) still exist in the root directory but are not tracked by git. You can:

- **Keep them** temporarily for reference
- **Delete them** once you've verified the new structure works:
  ```bash
  rm settings.py model_loader.py msat_evolver_manifold.py msat_evolver_helpers_manifold.py
  ```

## Questions?

If you encounter issues during migration, check:
1. Python path includes project root
2. All dependencies are installed (`pip install -r requirements.txt`)
3. Import statements use new module paths
4. Data paths in `config/settings.py` are correct
