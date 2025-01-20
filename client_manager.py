import flwr as fl
import requests
from typing import Dict, List, Optional
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from flwr.common.logger import log
import logging

class RandomOrgClientManager(SimpleClientManager):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[fl.server.criterion.Criterion] = None,
    ) -> List[ClientProxy]:
        log(logging.INFO, "Sampling clients using Random.org API")
        log(logging.INFO, f"Number of clients: {num_clients}")
        log(logging.INFO, f"Min number of clients: {min_num_clients}")
        """Sample a subset of available clients."""
        all_clients: Dict[str, ClientProxy] = self.clients
        if num_clients > len(all_clients):
            num_clients = len(all_clients)
        
        client_ids = list(all_clients.keys())
        
        sampled_indices = self._get_random_integers(0, len(client_ids) - 1, num_clients)
        
        if sampled_indices is None:
            # Fallback to default sampling if random.org fails
            return super().sample(num_clients, min_num_clients, criterion)
        
        sampled_clients: List[ClientProxy] = []
        for idx in sampled_indices:
            client = all_clients[client_ids[idx]]
            sampled_clients.append(client)
        
        log(logging.INFO, f"Sampled clients: {[client.cid for client in sampled_clients]}")
        return sampled_clients

    def _get_random_integers(self, min_value: int, max_value: int, count: int) -> Optional[List[int]]:
        url = "https://api.random.org/json-rpc/4/invoke"
        
        payload = {
            "jsonrpc": "2.0",
            "method": "generateIntegers",
            "params": {
                "apiKey": self.api_key,
                "n": count,
                "min": min_value,
                "max": max_value,
                "replacement": False,
                "base": 10
            },
            "id": 1
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            if "result" in data and "random" in data["result"]:
                return data["result"]["random"]["data"]
            else:
                raise ValueError("Unexpected response format")
        
        except requests.RequestException as e:
            print(f"Error making API request: {e}")
            return None
        except ValueError as e:
            print(f"Error processing response: {e}")
            return None


# # Example usage
# if __name__ == "__main__":
#     my_list = ["apple", "banana", "cherry", "date", "elderberry", "fig", "grape"]
#     sample_size = 4
    
#     result = sample_list_using_randomorg(my_list, sample_size)
#     if result:
#         print(f"Sampled items: {result}")
#     else:
#         print("Failed to sample the list")