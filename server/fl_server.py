"""
server/fl_server.py
Flower Federated Learning Server for FedMed-LLM
Runs FedAvg across 3 hospital clients for 5 rounds

Usage:
    python server/fl_server.py
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,
    Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# ── Config ────────────────────────────────────────────────────
NUM_ROUNDS      = 5
NUM_CLIENTS     = 3
MIN_FIT_CLIENTS = 3
SERVER_ADDRESS  = "0.0.0.0:8080"
LOG_DIR         = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class FedMedStrategy(FedAvg):
    """
    Custom FedAvg strategy that:
    - Logs metrics after every round
    - Saves round-by-round results to a JSON log
    - Passes the current round number to each client via config
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_history = []
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Pass round number to each client so they can log it."""
        config = {"server_round": server_round}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate weights and log per-round metrics."""

        # Call parent FedAvg aggregation
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            # Collect and log client losses
            client_losses = []
            client_samples = []
            for _, fit_res in results:
                loss = fit_res.metrics.get("train_loss", 0)
                n    = fit_res.num_examples
                client_losses.append(loss)
                client_samples.append(n)

            avg_loss = float(np.mean(client_losses))
            total_samples = sum(client_samples)

            round_log = {
                "round": server_round,
                "avg_train_loss": avg_loss,
                "client_losses": client_losses,
                "total_samples": total_samples,
            }
            self.round_history.append(round_log)

            print(f"\n{'='*55}")
            print(f"  FL Round {server_round}/{NUM_ROUNDS} complete")
            print(f"  Average training loss : {avg_loss:.4f}")
            print(f"  Total samples trained : {total_samples}")
            print(f"  Client losses         : {[f'{l:.4f}' for l in client_losses]}")
            print(f"{'='*55}")

            # Save aggregated weights after each round
            weights = parameters_to_ndarrays(aggregated_params)
            checkpoint_path = os.path.join(
                LOG_DIR, f"round_{server_round}_weights.npy"
            )
            np.save(checkpoint_path, np.array(weights, dtype=object))
            print(f"  Checkpoint saved → {checkpoint_path}")

            # Save metrics log
            log_path = os.path.join(LOG_DIR, f"training_log_{self.start_time}.json")
            with open(log_path, "w") as f:
                json.dump(self.round_history, f, indent=2)

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Log evaluation results."""
        if not results:
            return None, {}

        eval_losses = [res.loss for _, res in results]
        avg_eval_loss = float(np.mean(eval_losses))
        print(f"  Avg eval loss (round {server_round}): {avg_eval_loss:.4f}")

        return super().aggregate_evaluate(server_round, results, failures)


def main():
    print("\n" + "="*55)
    print("  FedMed-LLM Federated Learning Server")
    print(f"  Rounds: {NUM_ROUNDS}  |  Clients: {NUM_CLIENTS}")
    print(f"  Address: {SERVER_ADDRESS}")
    print("="*55 + "\n")

    strategy = FedMedStrategy(
        fraction_fit=1.0,           # Use all available clients each round
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_FIT_CLIENTS,
        min_available_clients=MIN_FIT_CLIENTS,
    )

    # Start the Flower server
    history = fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

    print(f"\nFederated training complete!")
    print(f"Training history saved in /logs/")
    print(f"\nNext step: run 'python scripts/merge_model.py' to merge LoRA weights into Phi-2")

    return history


if __name__ == "__main__":
    main()
