import argparse
import datetime


# import torch
# import wandb
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from ddpm import script_utils




def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
defaults = dict(
    learning_rate=2e-4,
    batch_size=128,
    iterations=800000,

    log_to_wandb=True,
    log_rate=1000,
    checkpoint_rate=1000,
    log_dir="~/ddpm_logs",
    project_name=None,
    run_name=run_name,

    model_checkpoint=None,
    optim_checkpoint=None,

    schedule_low=1e-4,
    schedule_high=0.02,

)
# print(defaults)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults

defaults.update(diffusion_defaults())
    # print(defaults)
parser = argparse.ArgumentParser()
for k, v in defaults.items():
    v_type = type(v)
    if v is None:
        v_type = str
    elif isinstance(v, bool):
        v_type = str2bool
    parser.add_argument(f"--{k}", default=v, type=v_type)

print(parser)