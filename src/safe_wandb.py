import os
import wandb
import numpy as np
from datetime import datetime

class WandbWrapper:
    """Mimics wandb but falls back to local logging if disabled."""
    def __init__(self, config, use_wandb=True, project="default_project"):
        self.active = use_wandb
        self.run = None

        # create local logging dir always
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.local_dir = os.path.join("runs", timestamp)
        os.makedirs(self.local_dir, exist_ok=True)

        if self.active:
            try:
                self.run = wandb.init(
                    project=project,
                    config=config,
                    dir=self.local_dir,
                    save_code=True,
                )
                print(f"[wandb] Online run: {self.run.name}")
            except Exception as e:
                print(f"[wandb] Could not initialize online run ({e}), switching to offline mode.")
                self.active = False

        if not self.active:
            print(f"[wandb] Offline mode — logs and checkpoints will be saved locally to {self.local_dir}")

    # ---- universal interface ----
    def log(self, *args, **kwargs):
        if self.active:
            wandb.log(*args, **kwargs)
        else:
            pass

    def save(self, filepath):
        """Save checkpoints even if W&B is off."""
        target_path = os.path.join(self.local_dir, os.path.basename(filepath))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        os.replace(filepath, target_path)
        if self.active:
            wandb.save(target_path)
        print(f"[wandb-wrapper] Saved file to {target_path}")

    def finish(self):
        if self.active:
            wandb.finish()
        print(f"[wandb-wrapper] Finished run — logs in {self.local_dir}")

    def log_image(self, tag, image, caption=""):
        if self.active:
            import wandb
            self.log({tag: [wandb.Image(image, caption=caption)]})
        else:
            # optionally save locally
            safe_tag = tag.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(self.local_dir, f"{safe_tag}.png")

            # Handle matplotlib Figure
        if hasattr(image, "savefig"):
            image.savefig(out_path, bbox_inches='tight')
        # Handle numpy arrays (float or uint8)
        elif isinstance(image, np.ndarray):
            from PIL import Image
            img = Image.fromarray(
                (255 * np.clip(image, 0, 1)).astype(np.uint8)
                if image.dtype != np.uint8 else image
            )
            img.save(out_path)
        else:
            raise TypeError(f"Unsupported image type for saving: {type(image)}")

        print(f"[wandb-wrapper] Saved image locally: {out_path}")

    def log_summary(self, metrics):
        if self.active and self.run is not None:
            for k, v in metrics.items():
                self.run.summary[k] = v
        else:
            with open(os.path.join(self.local_dir, "summary.txt"), "a") as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
