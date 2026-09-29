# Relay-D — Architecture

This diagram shows the end-to-end Learning-from-Demonstration (LfD) loop that
Relay-D supports, and how its two subsystems — `relay_d.acquisition` and
`relay_d.dispatch` — fit into it.

## Pipeline flowchart

```mermaid
flowchart TD
    A[YAML config] --> B["RelayDApp — Record\n(relay_d.acquisition)"]
    B --> C[Raw .h5 recording]
    C --> D["RelayDApp — Post-process\n(relay_d.acquisition)"]
    D --> E[Robomimic .h5 dataset]
    E --> F[Train policy\n external, e.g. robomimic]
    F --> G[Trained checkpoint .pth]

    H[obs_config.yaml] --> I
    J[action_config.yaml] --> I
    G --> I["lfd-inference-node\n(relay_d.dispatch)"]
    I --> K[Robot ROS 2 interfaces\n topics / actions]

    D -. shares action_config.yaml format via .-> L[DispatcherHelpers]
    L -. used by .-> I
```

## Subsystem relationship

- **`relay_d.acquisition`** owns recording (Config / Record / Post-process
  pages of the `RelayDApp` GUI, plus the headless `AppAPI`) and produces both
  the raw and Robomimic-format `.h5` datasets used to train a policy.
- **`relay_d.dispatch`** owns inference (the `lfd-inference-node` CLI and the
  `InferenceSession` API): it loads a trained checkpoint, builds observations
  from live sensor/TF data, runs the policy, and dispatches actions back to
  the robot.
- The two subsystems interoperate through `action_config.yaml`:
  `relay_d.acquisition`'s post-process page uses
  `relay_d.dispatch.DispatcherHelpers` so the same config format is produced
  during data prep and consumed at inference time.
- **`relay_d.utils`** provides small shared utilities (e.g. the colored
  console logger) used by both subsystems.

See the main [README](../../README.md) for installation and usage details.
