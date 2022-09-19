
from CenterDetector import CenterResNet
from CenterData import CenterPointDataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
def train(model, trainloader, testloader,optimizer, criterion, epochs):
    """
    Train the model given:
    - model: the model to train.
    - trainloader: the dataloader for the training set.
    - testloader: the dataloader for the test set.
    - optimizer: the optimizer to use.
    - criterion: the criterion to use.
    - epochs: the number of epochs to train for.
    """
    loss_history = []
    test_loss = []
    test_loss_indicies = []
    model.train()
    model.to(model.device)
    i = 0
    prev_iter = 0
    for epoch in range(epochs):
        for (img, target) in tqdm.tqdm(trainloader):
            i += 1
            img = img.to(model.device)
            target = target.to(model.device)
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()
            
        if epoch%5==0:
            
            test_loss.append(test(model,testloader,criterion))
            test_loss_indicies.append(i)
            if len(test_loss[:-1])==0 or test_loss[-1] < min(test_loss[:-1]):
                save_model(model, f"../CenterModels/model_best.pt")
            else:
                save_model(model, f"../CenterModels/model_last.pt")
            plt.plot(range(i),loss_history)
            plt.plot(test_loss_indicies,test_loss)
            plt.legend(["Training Loss", "Test Loss"])
            
            #plt.show()
            plt.savefig(f"../CenterModels/loss_history.png")
            plt.clf()
            print(f"Epoch {epoch} | Iteration {i} | Train Loss: {loss.item()} | Test Loss: {test_loss[-1]}")
    return loss_history,model
def l2_loss():
    """
    Compute the L2 loss between the predicction and target.
    """
    return nn.MSELoss()
def test(model, dataloader, criterion):
    """
    Test the model.
    """
    
    model.eval()
    loss = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(tqdm.tqdm(dataloader)):
            img = img.to(model.device)
            target = target.to(model.device)
            output = model(img)
            loss += criterion(output, target).item()
    return loss/len(dataloader)
def save_model(model, path):
    """
    Save the model.
    """
    torch.save(model.state_dict(), path)
def load_model(model, path):
    """
    Load the model.
    """
    model.load_state_dict(torch.load(path))
    return model
def create_model(model_path):
    """
    Create the model.
    """
    model = CenterResNet(img_shape=(1,3,320,320))
    model = load_model(model, model_path)
    return model
def main():
    """
    Main function.
    """
    dest_path = "../dataset/CenterData/"
    source_path = "../Xr-Synthesize-SRR-3/train/"
    image_folder = "images/"
    label_folder = "labels/"
    lr = 0.001
    batch_size = 64
    epochs = 20
    wd = 0.0001 
    print(f"Starting training with learning rate {lr}")
    # Create the model.
    model = CenterResNet(img_shape=(1,3,320,320))
    # Create the dataloader.
    dataset = CenterPointDataset(source_path=dest_path,image_folder=image_folder,label_folder=label_folder)
    trainloader, testloader = dataset.create_dataloader(batch_size=batch_size,shuffle=True,num_workers=1)
    # Create the optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
    # Create the criterion.
    criterion = l2_loss()
    # Train the model.
    loss_history, model = train(model, trainloader,testloader, optimizer, criterion, epochs=epochs)
    # Test the model.
    test_loss = test(model, testloader, criterion)
    print(f"Test loss: {test_loss}")
    # Save the model.
    date = time.strftime("%Y%m%d-%H%M%S")
    
    save_model(model, f"../CenterModels/model_{date}.pt")
    # Load the model.
    # Test the model.
    # Show the loss history.
    plt.plot(loss_history)
    plt.show()
def draw_circle(img, center, radius=2, color=(255,255,255), thickness=-1):
    """
    Draw a circle on an image.
    """
    print("Image shape: ",img.shape)
    print("Center: ",center)
    print("Radius: ",radius)
    print("Color: ",color)
    print("Thickness: ",thickness)

    cv2.circle(img, center, radius, color, thickness)
    return img
def test_on_image(model,testloader):
    """
    Run model on first batch in test loader and plot results to display:
    """
    model.eval().to(model.device)
    with torch.no_grad():
        (img, target) = next(iter(testloader))
        print(img)
        img = img.to(model.device)
        target = target.to(model.device)
        output = model(img)
        output = output.cpu().numpy()
        target = target.cpu().numpy()
        output = output.squeeze()
        target = target.squeeze()
        img_id = 0
        while img_id < len(img):
            next_image = False
            img[img_id]= draw_circle(img[img_id].cpu().numpy().astype(np.uint8), (int(output[img_id,0]*img.shape[-1]),int(output[img_id,1]*img.shape[-1])))
            print(f"First Circle Drawn")
            img[img_id]= draw_circle(img[img_id].astype(np.uint8), (int(target[img_id,0]),int(target[img_id,1])), color = (173,216,230))
            while not next_image:
                cv2.imshow("Output",img[img_id])
                key = cv2.waitKey(1)
                if key == ord('n'):
                    next_image = True
            img_id += 1
        cv2.destroyAllWindows()
        

def test_model(model_path):
    """
    Test the model.
    """
    dest_path = "../dataset/CenterData/"
    source_path = "../Xr-Synthesize-SRR-3/train/"
    image_folder = "images/"
    label_folder = "labels/"
    batch_size = 16
    model = CenterResNet(img_shape=(1,3,320,320))
    model = load_model(model, model_path)
    dataset = CenterPointDataset(source_path=dest_path,image_folder=image_folder,label_folder=label_folder)
    _,testloader = dataset.create_dataloader(batch_size=batch_size,shuffle=True,num_workers=1)
    test_on_image(model,testloader)


if __name__ == "__main__":
    #test_model("../CenterModels/model_best.pt")
    main()