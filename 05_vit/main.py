import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from tqdm import tqdm


def validate_model(model, dataloader, device, max_samples=None):
    """Validate the model on the dataset."""
    model.eval()
    correct = 0
    total = 0
    processed_samples = 0

    total_samples = len(dataloader.dataset)
    if max_samples:
        total_samples = min(total_samples, max_samples)

    with torch.no_grad():
        with tqdm(total=total_samples, desc="Validating", unit="samples") as pbar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                processed_samples += targets.size(0)
                accuracy = 100.0 * correct / total

                # Update the progress bar
                pbar.update(targets.size(0))
                pbar.set_postfix_str(f"Accuracy: {accuracy:.2f}%")

                if max_samples and processed_samples >= max_samples:
                    break

    return accuracy


if __name__ == "__main__":
    # Path to the ImageNet dataset
    imagenet_path = "/usr/scratch/sassauna1/ml_datasets/ILSVRC2012/"

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transformations for the dataset
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Hyperparameter: Maximum samples to validate
    max_samples = 1000  # Set to None for full validation
    subset_size = 50

    # Load the ImageNet validation dataset
    val_dataset = datasets.ImageFolder(root=f"{imagenet_path}/val", transform=transform)
    val_dataset = torch.utils.data.Subset(val_dataset, list(range(subset_size)))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Load a pre-trained small Vision Transformer (ViT)
    model = vit_b_16(weights="IMAGENET1K_V1")  # Using a pre-trained small ViT
    model = model.to(device)

    # Validate the model
    # print("Validating the model on ImageNet...")
    # accuracy = validate_model(model, val_loader, device, max_samples=max_samples)
    # print(f"Validation Accuracy: {accuracy:.2f}%")

    import copy
    from brevitas.graph.quantize import preprocess_for_quantize
    from brevitas.graph.quantize import quantize
    from brevitas.fx import brevitas_symbolic_trace

    model.to("cpu")
    fx_model = brevitas_symbolic_trace(model)

    # model = preprocess_for_quantize(model, equalize_iters=20, equalize_scale_computation="range")

    # # print(model)
    # # print(model.graph.print_tabular())

    # from torch import nn
    # import brevitas.nn as qnn
    # from brevitas.quant import Int8ActPerTensorFloat
    # from brevitas.graph.calibrate import calibration_mode

    # # from brevitas.quant import Int8ActPerTensorFloatMinMaxInit
    # from brevitas.quant import Int8WeightPerTensorFloat
    # from brevitas.quant import Int32Bias
    # from brevitas.quant import Uint8ActPerTensorFloat

    # # from brevitas.quant import Uint8ActPerTensorFloatMaxInit

    # COMPUTE_LAYER_MAP = {
    #     nn.AvgPool2d: (qnn.TruncAvgPool2d, {"return_quant_tensor": True}),
    #     nn.Conv2d: (
    #         qnn.QuantConv2d,
    #         {
    #             # 'input_quant': Int8ActPerTensorFloat,
    #             "weight_quant": Int8WeightPerTensorFloat,
    #             "output_quant": Int8ActPerTensorFloat,
    #             "bias_quant": Int32Bias,
    #             "return_quant_tensor": True,
    #             # 'input_bit_width': 8,
    #             "output_bit_width": 8,
    #             "weight_bit_width": 8,
    #         },
    #     ),
    #     nn.Linear: (
    #         qnn.QuantLinear,
    #         {
    #             # 'input_quant': Int8ActPerTensorFloat,
    #             "weight_quant": Int8WeightPerTensorFloat,
    #             "output_quant": Int8ActPerTensorFloat,
    #             "bias_quant": Int32Bias,
    #             "return_quant_tensor": True,
    #             # 'input_bit_width': 8,
    #             "output_bit_width": 8,
    #             "weight_bit_width": 8,
    #         },
    #     ),
    # }

    # QUANT_ACT_MAP = {
    #     nn.ReLU: (
    #         qnn.QuantReLU,
    #         {
    #             # 'input_quant': Int8ActPerTensorFloat,
    #             # 'input_bit_width': 8,
    #             "act_quant": Uint8ActPerTensorFloat,
    #             "return_quant_tensor": True,
    #             "bit_width": 7,
    #         },
    #     ),
    # }

    # QUANT_IDENTITY_MAP = {
    #     "signed": (qnn.QuantIdentity, {"act_quant": Int8ActPerTensorFloat, "return_quant_tensor": True, "bit_width": 7}),
    #     "unsigned": (qnn.QuantIdentity, {"act_quant": Uint8ActPerTensorFloat, "return_quant_tensor": True, "bit_width": 7}),
    # }

    # model_quant = quantize(
    #     copy.deepcopy(model),
    #     compute_layer_map=COMPUTE_LAYER_MAP,
    #     quant_act_map=QUANT_ACT_MAP,
    #     quant_identity_map=QUANT_IDENTITY_MAP,
    # )

    # def calibrate_model(model, calib_loader, device):
    #     model.eval()
    #     model.to(device)
    #     with torch.no_grad(), calibration_mode(model), tqdm(calib_loader, desc="Calibrating") as pbar:
    #         for images, _ in pbar:
    #             images = images.to(device)
    #             images = images.to(torch.float)
    #             model(images)

    # model_quant.eval()
    # model_quant = model_quant.to("cuda")
    # calibrate_model(model_quant, val_loader, "cuda")

    # val_dataset = datasets.ImageFolder(root=f"{imagenet_path}/val", transform=transform)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # print("Validating the model on ImageNet...")
    # accuracy = validate_model(model_quant, val_loader, "cuda", max_samples=max_samples)
    # print(f"Validation Accuracy: {accuracy:.2f}%")

    import IPython

    IPython.embed()
