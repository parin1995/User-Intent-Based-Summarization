import warnings
import wandb

__all__ = [
    "Logger",
    "WandBLogger"
]


class Logger:
    def __init__(self):
        self._log_dict = {}

    def collect(self, d: dict) -> dict:
        """
        Collect values to log upon commit.
        :param d: dictionary of metrics to collect
        :return: dictionary of currently connected metrics
        """
        for k, v in d.items():
            if k in self._log_dict and self._log_dict[k] != v:
                warnings.warn(
                    f"Logger had uncommitted value `{k}={self._log_dict[k]}`, overwriting with `{k}={v}`."
                )
            self._log_dict[k] = v
        return self._log_dict

    def commit(self) -> None:
        """
        If there are values to log, log them.
        """
        self.clear()

    def clear(self) -> None:
        """
        Clear collected log values.
        """
        self._log_dict = {}

    @property
    def has_collected_data(self) -> bool:
        return len(self._log_dict) > 0


class WandBLogger(Logger):
    def commit(self) -> None:
        """
        If there are values to log, log them.
        """
        if self.has_collected_data:
            wandb.log(self._log_dict)
            self.clear()
