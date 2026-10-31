

def model_config():
    config = {
        # --- student models ---
        "ti": { # tiny model
            "model_name": "vit-ti",
            "model_dims": 128,
            "num_heads": 2,
            "depth": 2,
        }, # 0.5M params
        
        "mi": { # mini model
            "model_name": "vit-mi",
            "model_dims": 256,
            "num_heads": 4,
            "depth": 4,
        }, # 3.36 M params

        "s": { # small model
            "model_name": "vit-s",
            "model_dims": 384,
            "num_heads": 6,
            "depth": 6,
        }, # 10.94 M params

        "m": { # medium model
            "model_name": "vit-m",
            "model_dims": 512,
            "num_heads": 8,
            "depth": 8,
        }, # 25 M params

        "b": { # big model
            "model_name": "vit-b",
            "model_dims": 768,
            "num_heads": 12,
            "depth": 12,
        }, # 85.65 M params

        "l": { # large model
            "model_name": "vit-l",
            "model_dims": 1024,
            "num_heads": 16,
            "depth": 16,
        }, # 202 M params

        "xl": { # extra large model
            "model_name": "vit-xl",
            "model_dims": 1536,
            "num_heads": 24,
            "depth": 20,
        }, # 567 M params

        "xxl": { # extra extra large model
            "model_name": "vit-xxl",
            "model_dims": 2048,
            "num_heads": 32,
            "depth": 24,
        }, # 1.2 B params
    }
    return config