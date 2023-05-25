import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from timm.models import create_model
class Cluster():
    def __init__(self,device):
        self.all_keys=[]
        print(f"Creating model for image feature extractor")
        self.image_encoder=create_model('vit_base_patch16_224', pretrained=True)
        self.image_encoder.to(device)
        
    def extract_vector(self, image):
        image_features = self.image_encoder(image)['x']#forward ViT
        #print(f"image_features:{image_features}")
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def clustering(self, dataloader, device: torch.device): # after training 1 task, cluster image features
        features = []
        for inputs, targets, domain, class_group in dataloader:
            inputs, targets = inputs.to(device,non_blocking=True), targets.to(device,non_blocking=True)
            #mask = (targets >= self._known_classes).nonzero().view(-1) #! ???
            #inputs = torch.index_select(inputs, 0, mask)
            with torch.no_grad():
                if isinstance(self, torch.nn.DataParallel):
                    feature = self.module.extract_vector(inputs)
                else:
                    feature = self.extract_vector(inputs)
            feature = feature.mean(dim=1)  # (bs, embed_dim)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature)
        features = torch.cat(features, 0).cpu().detach().numpy()
        
        clustering = KMeans(n_clusters=5, random_state=0).fit(features)
        self.all_keys.append(torch.tensor(clustering.cluster_centers_).to(feature.device))
        #print("Clustering finished")
    
    def task_selection(self, inputs):
        with torch.no_grad():
                if isinstance(self, nn.DataParallel):
                    feature = self.module.extract_vector(inputs)
                else:
                    feature = self.extract_vector(inputs)
                #print(f"feature.shape: {feature.shape}") #[bs, 512]

                taskselection = []
                for task_centers in self.all_keys: #! domain centers
                    tmpcentersbatch = []
                    for center in task_centers: #! ??????
                        tmpcentersbatch.append((((feature - center) ** 2) ** 0.5).sum(1))
                    taskselection.append(torch.vstack(tmpcentersbatch).min(0)[0]) 
                #print(f"taskselection.shape:{len(taskselection)}") #1
                selection = torch.vstack(taskselection).min(0)[1] #(bs)
        return selection
    
    