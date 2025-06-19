"""Class registry for models, layers, optimizers, and schedulers.

"""

optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Downstream task embedding backbones
    "dna_embedding": "src.models.sequence.dna_embedding.DNAEmbeddingModel",
    "dna_embedding_caduceus": "src.models.sequence.dna_embedding.DNAEmbeddingModelCustom",
    'dna_embedding_nt':'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_ntv2':'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_genalm':'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_dnabert':'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_dnabert2':'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_dnaberts':'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_grover': 'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    'dna_embedding_gpn': 'src.models.sequence.dna_embedding.DNAEmbeddingModelCustom',
    "dna_embedding_splicebert": "src.models.sequence.dna_embedding.DNAEmbeddingModelCustom",
    "dna_embedding_rnafm": "src.models.sequence.dna_embedding.DNAEmbeddingModelCustom",
    "dna_embedding_rnaernie": "src.models.sequence.dna_embedding.DNAEmbeddingModelCustom",
    "dna_embedding_mistraldna": "src.models.sequence.dna_embedding.DNAEmbeddingModelCustom",
    "dna_embedding_generator": "src.models.sequence.dna_embedding.DNAEmbeddingModelCustom",
    
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "ff": "src.models.sequence.ff.FF",
    "hyena": "src.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "src.models.sequence.hyena.HyenaFilter",
}

callbacks = {
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_every_n_steps": "pytorch_lightning.callbacks.ModelCheckpoint",
    "model_checkpoint_last": "pytorch_lightning.callbacks.ModelCheckpoint",  
    "model_checkpoint_every_epoch": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "params": "src.callbacks.params.ParamsLog",
    "timer": "src.callbacks.timer.Timer",
    "val_every_n_global_steps": "src.callbacks.validation.ValEveryNGlobalSteps",
}

model_state_hook = {
    'load_backbone': 'src.models.sequence.dna_embedding.load_backbone',
}


layer_decay = {
    "layer_decay": "src.utils.layer_decay.get_layer_id",
}
