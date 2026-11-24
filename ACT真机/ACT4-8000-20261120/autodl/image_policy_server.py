"""Strict LeRobot ACT server for locally prepared 224 RGB observations."""

from concurrent import futures
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
import time

import draccus
import grpc
import numpy as np
import torch

from image_preprocessing import validate_model_contract
from episode_policy_guard import install_episode_guard
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedObservation,
    extract_state_from_raw_observation,
    make_lerobot_observation,
    prepare_image,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc


CONTRACT_ATTRIBUTE = "image_preprocessing_contract"


@dataclass
class ImagePolicyServerConfig(PolicyServerConfig):
    model_path: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not self.model_path:
            raise ValueError("model_path must name the 224 ACT checkpoint to serve.")


def validate_policy_specs(specs, model_path, contract, image_keys):
    """Reject incompatible clients before loading or replacing the model."""
    if not isinstance(specs, RemotePolicyConfig):
        raise TypeError("Expected a LeRobot RemotePolicyConfig.")
    if specs.policy_type != "act":
        raise ValueError("This server supports the configured ACT model.")
    if Path(specs.pretrained_name_or_path).resolve() != model_path:
        raise ValueError("Requested checkpoint differs from this server's configured model_path.")
    if getattr(specs, CONTRACT_ATTRIBUTE, None) != contract:
        raise ValueError("Client and model image preprocessing contracts differ.")
    cameras = {
        key: feature for key, feature in specs.lerobot_features.items()
        if key.startswith("observation.images.")
    }
    renamed_keys = {specs.rename_map.get(key, key) for key in cameras}
    if renamed_keys != image_keys or len(cameras) != len(image_keys):
        raise ValueError("Client camera names differ from the checkpoint image features.")
    for key, feature in cameras.items():
        if (feature.get("dtype") not in {"image", "video"}
                or list(feature.get("shape", [])) != [224, 224, 3]
                or feature.get("names") != ["height", "width", "channels"]
                or feature.get("info", {}).get("is_depth_map")):
            raise ValueError(f"Client camera feature must declare RGB HWC 224 for {key}.")


def validate_transport_observation(observation, image_features):
    if not isinstance(observation, TimedObservation):
        raise TypeError("Expected a LeRobot TimedObservation.")
    values = observation.get_observation()
    for key in image_features:
        camera_name = key.removeprefix("observation.images.")
        image = values.get(camera_name)
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != (224, 224, 3)):
            raise ValueError(f"Camera {camera_name} must arrive as RGB uint8 HWC 224.")


def observation_without_resize(observation, features):
    """Use LeRobot feature naming and float conversion with no image resampling."""
    validate_transport_observation(observation, {
        key: feature for key, feature in features.items()
        if key.startswith("observation.images.")
    })
    raw = observation.get_observation()
    values = make_lerobot_observation(raw, features)
    prepared = {"observation.state": extract_state_from_raw_observation(values)}
    for key, value in values.items():
        if key.startswith("observation.images."):
            image = torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1)
            prepared[key] = prepare_image(image).unsqueeze(0)
    if "task" in raw:
        prepared["task"] = raw["task"]
    return prepared


class ImagePolicyServer(PolicyServer):
    def __init__(self, config):
        self.model_path = Path(config.model_path).resolve()
        self.image_contract = validate_model_contract(self.model_path)
        model_config = json.loads((self.model_path / "config.json").read_text(encoding="utf-8-sig"))
        self.expected_image_keys = {
            key for key, feature in model_config["input_features"].items()
            if feature["type"] == "VISUAL"
        }
        self.contract_handshake_complete = False
        super().__init__(config)

    def Ready(self, request, context):
        self.contract_handshake_complete = False
        return super().Ready(request, context)

    def SendPolicyInstructions(self, request, context):
        self.contract_handshake_complete = False
        specs = pickle.loads(request.data)
        try:
            validate_policy_specs(
                specs, self.model_path, self.image_contract, self.expected_image_keys,
            )
        except (TypeError, ValueError, KeyError, AttributeError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        response = super().SendPolicyInstructions(request, context)
        self.contract_handshake_complete = self.policy is not None
        return response

    def GetActions(self, request, context):
        if not self.contract_handshake_complete:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Complete the matching image-contract handshake first.")
        return super().GetActions(request, context)

    def SendObservations(self, request_iterator, context):
        if not self.contract_handshake_complete:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Complete the matching image-contract handshake first.")
        try:
            return super().SendObservations(request_iterator, context)
        except (TypeError, ValueError, KeyError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))

    def _enqueue_observation(self, observation):
        validate_transport_observation(observation, {
            key: feature for key, feature in self.lerobot_features.items()
            if key.startswith("observation.images.")
        })
        return super()._enqueue_observation(observation)

    def _predict_action_chunk(self, observation):
        # LeRobot 0.6.1 hardcodes a resize inside this method's input conversion.
        # Retain its processors, ACT call and action timing, and replace that conversion.
        started = time.perf_counter()
        batch = observation_without_resize(observation, self.lerobot_features)
        batch = self.preprocessor(batch)
        self.last_processed_obs = observation
        actions = self._get_action_chunk(batch)
        processed = [self.postprocessor(actions[:, index, :]) for index in range(actions.shape[1])]
        actions = torch.stack(processed, dim=1).squeeze(0).detach().cpu()
        timed_actions = self._time_action_chunk(
            observation.get_timestamp(), list(actions), observation.get_timestep(),
        )
        self.logger.info(
            "Observation %s preprocessing and inference %.2f ms",
            observation.get_timestep(), (time.perf_counter() - started) * 1000,
        )
        return timed_actions


install_episode_guard(ImagePolicyServer)


@draccus.wrap()
def serve(cfg: ImagePolicyServerConfig):
    policy_server = ImagePolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    if server.add_insecure_port(f"{cfg.host}:{cfg.port}") == 0:
        raise RuntimeError("Could not bind the policy server address.")
    server.start()
    policy_server.logger.info("224 image-contract policy server started on %s:%s", cfg.host, cfg.port)
    try:
        server.wait_for_termination()
    finally:
        policy_server.stop()
        server.stop(1.0).wait()


if __name__ == "__main__":
    serve()
