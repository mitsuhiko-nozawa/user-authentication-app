from .base_model import BaseModel
from .networks import *
from utils_torch import run_training, inference_fn
#from .run_utils import run_training, inference_fn
from utils import seed_everything

from pathlib import Path


class FaceRecognitionModel(BaseModel):
    def fit(self, trainloader, validloader):
        run_training(
            model=self.model,
            trainloader=trainloader,
            validloader=validloader,
            epochs=self.epochs,
            optimizer=self.optimizer,
            optimizer_params=self.optimizer_params,
            scheduler=self.scheduler,
            scheduler_params=self.scheduler_params,
            loss_tr=self.loss_tr,
            loss_fn=self.loss_fn,
            early_stopping_steps=self.early_stopping_steps,
            verbose=self.verbose,
            device=self.device,
            seed=self.seed,
            fold=self.fold,
            weight_path=self.weight_path,
            mixing=self.params["mixing"],
            log_path=self.log_path,
            grad_accum_step=self.accum_iter,
        )
        
    def predict(self, testloader):
        embs = inference_fn(model=self.model, dataloader=testloader, device=self.device)
        return embs

    def read_weight(self, fname):
        fname = f"{self.seed}_{self.fold}.pt"
        # networkクラスの名前でコンフリクトする
        #print(torch.load( Path(self.weight_path) / fname , map_location=self.device))
        
        self.model.model.load_state_dict(torch.load( Path(self.weight_path) / fname , map_location=self.device), self.device)
        
        
    def save_weight(self):
        pass
    
    def get_model(self, model_name):
        model = eval(model_name)(n_classes=self.n_classes)
        model.to(self.device)
        return model
