#!/usr/bin/env python
'''

"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'atmospheric_gases.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

'''

#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    """
    Run administrative tasks.

    This file lives in:   <project_root>/Backend/manage.py
    We also have ML code in: <project_root>/Data/...

    To allow imports like:
        from Data.ExtractData_IAModel.multivar_forecast import ...
    we add <project_root> to sys.path before starting Django.
    """
    # Backend directory: .../AeroSense/Backend
    backend_dir = Path(__file__).resolve().parent
    # Project root:      .../AeroSense
    project_root = backend_dir.parent

    # Ensure project root is on PYTHONPATH
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Standard Django bootstrap
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "atmospheric_gases.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
