import zarr
import matplotlib.pyplot as plt

import zarr
import matplotlib.pyplot as plt
import ipywidgets as widgets
from pathlib import Path

class ZarrStore:
    def __init__(self, path: str, run_names: list[str]):
        self.path = Path(path)
        self.run_names = run_names
        self.all_run_names = run_names
        self._load_stores(run_names)

    def _load_stores(self, run_names: list[str]):
        """Helper to open Zarr stores for the currently active runs."""
        self._stores = {
            run_name: zarr.open(str(self.path / run_name), mode='r')
            for run_name in run_names
        }

    @staticmethod
    def create(path: str):
        def _discover_runs(path) -> list[str]:
            return sorted([d.name for d in path.iterdir() if d.is_dir()], reverse=True)
        return ZarrStore(path, run_names=_discover_runs(Path(path)))

    def length(self, run_name: str, metric_name: str) -> int:
        return self._stores[run_name][f'{metric_name}/steps'].shape[0] # type: ignore

    def load_metric(self, metric_name: str, run_names: list[str] | None = None, index: int | None = None) -> dict:
        run_names = run_names or self.run_names
        return {
            run_name: self.load_metric_for(run_name, metric_name, index=index)
            for run_name in run_names
        }

    def load_metric_for(self, run_name: str, metric_name: str, index: int | None = None) -> tuple:
        if index is None:
            steps = self._stores[run_name][f'{metric_name}/steps'][:] # type: ignore
            values = self._stores[run_name][f'{metric_name}/values'][:] # type: ignore
        else:
            steps = self._stores[run_name][f'{metric_name}/steps'][index] # type: ignore
            values = self._stores[run_name][f'{metric_name}/values'][index] # type: ignore
        return steps, values

    def create_ui(self):
        """Generates and returns the interactive ipywidgets UI."""
        # 1. Create checkboxes
        checkboxes = [
            widgets.Checkbox(
                value=(run in self.run_names), 
                description=run
            ) 
            for run in self.all_run_names
        ]

        output = widgets.Output()

        # 2. Define the reactive function
        def on_checkbox_change(change):
            with output:
                output.clear_output()
                self.run_names = [cb.description for cb in checkboxes if cb.value]

        # 3. Attach the observer to EVERY checkbox
        # The names='value' argument tells it to only fire when the checkbox is toggled
        for cb in checkboxes:
            cb.observe(on_checkbox_change, names='value')

        # 4. Return the UI without the button
        ui = widgets.VBox([widgets.Label(value="Select runs to load:")] + checkboxes + [output])
        return ui
    